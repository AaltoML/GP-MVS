import argparse

import numpy as np
from path import Path
import pylab as plt
from scipy.linalg import expm

import cv2

from numpy.linalg import inv
from tqdm import tqdm

import os

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch import Tensor

from enCoder import enCoder
from deCoder import deCoder
from GPlayer import GPlayer


parser = argparse.ArgumentParser(description='Multi-view depth estimation',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('seqpath', metavar='DIR',
                    help='path to formatted seq')
parser.add_argument('--savepath',  default=None,
                    help='save path of predictions, None means will not save' )

parser.add_argument('--encoder', default='encoder_model_best.pth.tar',
                     help='path to pretrained encoder model')
parser.add_argument('--gp', default='gp_model_best.pth.tar',
                     help='path to pretrained gp model')
parser.add_argument('--decoder', default='decoder_model_best.pth.tar',
                     help='path to pretrained decoder model')

args = parser.parse_args()



def genDistM(poses):
    n = len(poses)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = pose_distance(poses[i], poses[j])
    return D


def pose_distance(p1, p2):
    rel_pose = np.dot(p1, inv(p2))
    R = rel_pose[:3, :3]
    t = rel_pose[:3, 3]

    return round(np.sqrt(np.linalg.norm(t) ** 2 + 2 * (1 - min(3.0, np.matrix.trace(R)) / 3)), 4)


def compute_errors(gt, pred):
    valid1 = gt > 0.5
    valid2 = gt < 50
    valid = valid1 & valid2

    gt = gt[valid]
    pred = 1 / pred[valid]

    L1 = np.mean(np.abs(gt - pred))
    L1_rel = np.mean(np.abs(gt - pred) / gt)
    L1_inv = np.mean(np.abs(1 / gt - 1 / pred))

    log_diff = np.log(gt) - np.log(pred)
    sc_inv = np.sqrt(np.mean(np.square(log_diff)) - np.square(np.mean(log_diff)))

    return L1, L1_rel, L1_inv, sc_inv


pixel_coordinate = np.indices([320, 256]).astype(np.float32)
pixel_coordinate = np.concatenate(
    (pixel_coordinate, np.ones([1, 320, 256])), axis=0)
pixel_coordinate = np.reshape(pixel_coordinate, [3, -1])

def encoder_forward(r_img,n_img, r_pose,n_pose, K):

    left_image = r_img
    right_image = n_img


    left_pose = r_pose
    right_pose = n_pose

    camera_k = K


    left2right = np.dot(right_pose, inv(left_pose))

    # scale to 320x256
    original_width = left_image.shape[1]
    original_height = left_image.shape[0]
    factor_x = 320.0 / original_width
    factor_y = 256.0 / original_height

    left_image = cv2.resize(left_image, (320, 256))
    right_image = cv2.resize(right_image, (320, 256))
    camera_k[0, :] *= factor_x
    camera_k[1, :] *= factor_y

    # convert to torch
    torch_left_image = np.moveaxis(left_image, -1, 0)
    torch_left_image = np.expand_dims(torch_left_image, 0)

    torch_left_image = (torch_left_image - 81.0)/ 35.0
    torch_right_image = np.moveaxis(right_image, -1, 0)
    torch_right_image = np.expand_dims(torch_right_image, 0)

    torch_right_image = (torch_right_image - 81.0)/ 35.0


    left_image_cuda = Tensor(torch_left_image).cuda()
    left_image_cuda = Variable(left_image_cuda)

    right_image_cuda = Tensor(torch_right_image).cuda()
    right_image_cuda = Variable(right_image_cuda)

    left_in_right_T = left2right[0:3, 3]
    left_in_right_R = left2right[0:3, 0:3]
    K = camera_k
    K_inverse = inv(K)
    KRK_i = K.dot(left_in_right_R.dot(K_inverse))
    KRKiUV = KRK_i.dot(pixel_coordinate)
    KT = K.dot(left_in_right_T)
    KT = np.expand_dims(KT, -1)
    KT = np.expand_dims(KT, 0)
    KT = KT.astype(np.float32)
    KRKiUV = KRKiUV.astype(np.float32)
    KRKiUV = np.expand_dims(KRKiUV, 0)
    KRKiUV_cuda_T = Tensor(KRKiUV).cuda()
    KT_cuda_T = Tensor(KT).cuda()

    conv5, conv4, conv3, conv2, conv1= encoder(left_image_cuda, right_image_cuda, KRKiUV_cuda_T,KT_cuda_T)

    return conv5, conv4, conv3, conv2, conv1

#load formatted sequence
scene = Path(args.seqpath)
intrinsics = np.loadtxt(scene / 'K.txt').astype(np.float32).reshape((3, 3))
imgs = sorted((scene/'images').files('*.png'))
gts = sorted((scene/'depth').files('*.npy'))

gt_poses = []
with open(scene / 'poses.txt') as f:
    for l in f.readlines():
        l = l.strip('\n')
        gt_poses.append(np.array(l.split(' ')).astype(np.float32).reshape(4, 4))

#load pre-trained model
pretrained_encoder = args.encoder
pretrained_gplayer = args.gp
pretrained_decoder = args.decoder

encoder = enCoder().cuda()
encoder = torch.nn.DataParallel(encoder)
weights = torch.load(pretrained_encoder)
encoder.load_state_dict(weights['state_dict'])
encoder.eval()

decoder = deCoder().cuda()
decoder = torch.nn.DataParallel(decoder)
weights = torch.load(pretrained_decoder)
decoder.load_state_dict(weights['state_dict'])
decoder.eval()

gplayer =GPlayer()
weights = torch.load(pretrained_gplayer)
gplayer.load_state_dict(weights['state_dict'])
gplayer.eval()

# load values of hyperparameters
gamma2 = np.exp(weights['state_dict']['gamma2'][0].item())
ell = np.exp(weights['state_dict']['ell'][0].item())
sigma2 = np.exp(weights['state_dict']['sigma2'][0].item())


n = len(imgs)

distM = genDistM(gt_poses)

with torch.no_grad():
    poses = []
    idepths = []
    idepths_after = []
    latents = []
    conv1s = []
    conv2s = []
    conv3s = []
    conv4s = []

    preds = []

    lam = np.sqrt(3) / ell;
    F = np.array([[0, 1], [-lam ** 2, -2 * lam]])
    Pinf = np.array([[gamma2, 0], [0, gamma2 * lam ** 2]])
    h = np.array([[1], [0]])

    # State mean and covariance
    M = np.zeros((F.shape[0], 512 * 8 * 10))
    P = np.zeros((F.shape[0], F.shape[0]))
    P = Pinf



    depth_gts = []
    for i in tqdm(range(1, n)):  # start with the 2nd frame

        r_pose = gt_poses[i]
        n_pose = gt_poses[i - 1]

        r_img = cv2.imread(imgs[i])
        n_img = cv2.imread(imgs[i - 1])

        gt_depth = np.load(gts[i])
        depth_gts.append(gt_depth)

        camera_k = np.loadtxt(scene / 'K.txt').astype(np.float32).reshape((3, 3))

        conv5, conv4, conv3, conv2, conv1 = encoder_forward(r_img, n_img, r_pose, n_pose, camera_k)



        batch, channel, height, width = conv5.size()
        y = np.expand_dims(conv5.cpu().numpy().flatten(), axis=0)

        dt = distM[i, i - 1]
        A = expm(F * dt)
        Q = Pinf - A.dot(Pinf).dot(A.T)
        M = A.dot(M)
        P = A.dot(P).dot(A.T) + Q

        # Update step
        v = y - h.T.dot(M)
        s = h.T.dot(P).dot(h) + sigma2
        k = P.dot(h) / s
        M += k.dot(v)
        P -= k.dot(h.T).dot(P)

        Z = torch.from_numpy(M[0]).view(batch, channel, height, width).float().cuda()
        Z = torch.nn.functional.relu(Z)
        pred = decoder(Z, conv4, conv3, conv2, conv1)
        idepths.append(pred[0][0].cpu().data.numpy())


    error_names = ['L1', 'L1_rel', 'L1_inv', 'sc_inv']
    errors = np.zeros((1, len(error_names), len(idepths)))

    for i in range(n - 1):
        gt = depth_gts[i]
        h, w = gt.shape

        pred = cv2.resize(idepths[i], (w, h))
        pred = np.clip(pred, a_min=0.02, a_max=2)  # depth range within [0.5, 50]
        preds.append(pred)

        errors[0, :, i] = compute_errors(gt, pred)

mean_errors = errors.mean(2)
print("Results for original methods : ")
print("{:>10}, {:>10}, {:>10}, {:>10}".format(*error_names))
print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(*mean_errors[0]))

if args.savepath is not None:
    np.save(args.savepath, np.array(preds))