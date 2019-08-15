import argparse
import time

import torch.backends.cudnn as cudnn
import torch.optim

import torch.utils.data
import custom_transforms
from deCoder import deCoder
from enCoder import enCoder
from GPlayer import GPlayer
from utils import tensor2array, save_checkpoint, save_path_formatter


from loss_functions import *

from logger import TermLogger, AverageMeter
from itertools import chain
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='Multi-view depth estimation',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for training', default=3)

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch-size', default=0, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if not set)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameters for adam')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--pretrained-dict', dest='pretrained_dict', default=None, metavar='PATH',
                    help='path to pre-trained dispnet model')


parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH',
                    help='csv where to save per-epoch train and valid stats')
parser.add_argument('--log-full', default='progress_log_full.csv', metavar='PATH',
                    help='csv where to save per-gradient descent train stats')

parser.add_argument('--log-output', action='store_true', help='will log dispnet outputs and warped imgs at validation step')
parser.add_argument('-f', '--training-output-freq', type=int, help='frequence for outputting dispnet outputs and warped imgs at training for all scales if 0 will not output',
                    metavar='N', default=0)





best_error = -1
n_iter = 0
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def main():
    global args, best_error, n_iter, device
    args = parser.parse_args()
    from dataset_loader import SequenceFolder

    save_path = save_path_formatter(args, parser)
    args.save_path = 'checkpoints'/save_path
    print('=> will save everything to {}'.format(args.save_path))
    args.save_path.makedirs_p()
    torch.manual_seed(args.seed)

    training_writer = SummaryWriter(args.save_path)
    output_writers = []
    if args.log_output:
        for i in range(3):
            output_writers.append(SummaryWriter(args.save_path/'valid'/str(i)))

    train_transform = custom_transforms.Compose([
        custom_transforms.ArrayToTensor(),
    ])

    valid_transform = custom_transforms.Compose([
        custom_transforms.ArrayToTensor(),
    ])



    print("=> fetching scenes in '{}'".format(args.data))
    train_set = SequenceFolder(
        args.data,
        transform=train_transform,
        seed=args.seed,
        train=True,
        sequence_length=args.sequence_length
    )

    val_set = SequenceFolder(
            args.data,
            transform=valid_transform,
            seed=args.seed,
            train=False,
            sequence_length=args.sequence_length,
        )
    print('{} samples found in {} train scenes'.format(len(train_set), len(train_set.scenes)))
    print('{} samples found in {} valid scenes'.format(len(val_set), len(val_set.scenes)))
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)


    encoder = enCoder()
    decoder = deCoder()
    gplayer = GPlayer()

    if args.pretrained_dict:
        print("=> using pre-trained weights")
        weights = torch.load(args.pretrained_dict)
        pretrained_dict = weights['state_dict']

        encoder_dict = encoder.state_dict()
        pretrained_dict_encoder = {k: v for k, v in pretrained_dict.items() if k in encoder_dict}
        encoder_dict.update(pretrained_dict_encoder)
        encoder.load_state_dict(pretrained_dict_encoder)

        decoder_dict = decoder.state_dict()
        pretrained_dict_decoder = {k: v for k, v in pretrained_dict.items() if k in decoder_dict}
        decoder_dict.update(pretrained_dict_decoder)
        decoder.load_state_dict(pretrained_dict_decoder)

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    cudnn.benchmark = True
    encoder = torch.nn.DataParallel(encoder)
    decoder = torch.nn.DataParallel(decoder)

    parameters = chain(encoder.parameters(), gplayer.parameters(), decoder.parameters())
    optimizer = torch.optim.Adam(parameters, args.lr,
                                 betas=(args.momentum, args.beta),
                                 weight_decay=args.weight_decay)


    logger = TermLogger(n_epochs=args.epochs, train_size=min(len(train_loader), args.epoch_size), valid_size=len(val_loader))

    logger.epoch_bar.start()

    for epoch in range(args.epochs):
        logger.epoch_bar.update(epoch)

        # train for one epoch
        logger.reset_train_bar()

        train_loss = train(train_loader, encoder, gplayer, decoder, optimizer, args.epoch_size, logger, training_writer)
        logger.train_writer.write(' * Avg Loss : {:.3f}'.format(train_loss))

        # evaluate on validation set
        logger.reset_valid_bar()

        errors, error_names = validate(val_loader, encoder, gplayer, decoder, epoch, logger, output_writers)
        error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in zip(error_names, errors))
        logger.valid_writer.write(' * Avg {}'.format(error_string))

        for error, name in zip(errors, error_names):
            training_writer.add_scalar(name, error, epoch)


        decisive_error = errors[-1]
        if best_error < 0:
            best_error = decisive_error

        # save best checkpoint
        is_best = decisive_error < best_error
        best_error = min(best_error, decisive_error)
        save_checkpoint(
            args.save_path, {
                'epoch': epoch + 1,
                'state_dict': encoder.state_dict()
            },
            {
                'epoch': epoch + 1,
                'state_dict': gplayer.state_dict()
            },
             {
                 'epoch': epoch + 1,
                 'state_dict': decoder.state_dict()
             },
            is_best)

    logger.epoch_bar.finish()


def train(train_loader, encoder, gplayer, decoder, optimizer, epoch_size, logger, train_writer):

    global args, n_iter, device
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(precision=4)

    # switch to train mode
    encoder.train()
    decoder.train()
    gplayer.train()

    end = time.time()
    logger.train_bar.update(0)


    for i, (imgs, KRKiUVs, KTs, gts, D) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        imgs_var = [img.to(device) for img in imgs]
        gts_var = [gt.to(device) for gt in gts]

        KRKiUV_cuda_Ts = [KRKiUV.to(device) for KRKiUV in KRKiUVs]
        KT_cuda_Ts = [KT.to(device) for KT in KTs]

        latents = []
        conv4s = []
        conv3s = []
        conv2s = []
        conv1s = []

        loss = 0
        for k,img in enumerate(imgs_var[:-1]):
            r_img = imgs_var[k]
            n_img = imgs_var[k+1]
            KRKiUV_cuda_T = KRKiUV_cuda_Ts[k]
            KT_cuda_T = KT_cuda_Ts[k]
            conv5, conv4, conv3, conv2, conv1 = encoder(r_img, n_img, KRKiUV_cuda_T, KT_cuda_T)

            latents.append(conv5)
            conv4s.append(conv4)
            conv3s.append(conv3)
            conv2s.append(conv2)
            conv1s.append(conv1)

        #deal with the last frame that use the previous frame as neighbour
        r_img = imgs_var[-1]
        n_img = imgs_var[-2]
        KRKiUV_cuda_T = KRKiUV_cuda_Ts[-1]
        KT_cuda_T = KT_cuda_Ts[-1]
        conv5, conv4, conv3, conv2, conv1 = encoder(r_img, n_img, KRKiUV_cuda_T, KT_cuda_T)

        latents.append(conv5)
        conv4s.append(conv4)
        conv3s.append(conv3)
        conv2s.append(conv2)
        conv1s.append(conv1)

        Y = torch.stack(latents, dim = 1).cpu()

        Z = gplayer(D,Y)
        b,l,c,h,w = Y.size()

        for k,img in enumerate(imgs_var):
            conv5 = Z[:,k].view(b,c,h,w).cuda()
            conv4 = conv4s[k]
            conv3 = conv3s[k]
            conv2 = conv2s[k]
            conv1 = conv1s[k]
            pred = decoder(conv5, conv4, conv3, conv2, conv1)
            loss += compute_errors(gts_var[k],pred)

        loss = loss/len(imgs_var)

        if i > 0 and n_iter % args.print_freq == 0:
            train_writer.add_scalar('Total_loss', loss.item(), n_iter)

        # record loss
        losses.update(loss.item(), args.batch_size)

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        logger.train_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.train_writer.write('Train: Time {} Data {} Loss {}'.format(batch_time, data_time, losses))
        if i >= epoch_size - 1:
            break

        n_iter += 1

    return losses.avg[0]

def validate(val_loader, encoder, gplayer, decoder, epoch, logger, output_writers=[]):
    global args
    batch_time = AverageMeter()

    errors = AverageMeter(i=1, precision=4)
    log_outputs = len(output_writers) > 0

    # switch to evaluate mode
    encoder.eval()
    decoder.eval()
    gplayer.eval()


    end = time.time()
    logger.valid_bar.update(0)

    for i, (imgs, KRKiUVs, KTs, gts, D) in enumerate(val_loader):

        with torch.no_grad():

            imgs_var = [img.to(device) for img in imgs]
            gts_var = [gt.to(device) for gt in gts]

            KRKiUV_cuda_Ts = [KRKiUV.to(device) for KRKiUV in KRKiUVs]
            KT_cuda_Ts = [KT.to(device) for KT in KTs]

            latents = []
            conv4s = []
            conv3s = []
            conv2s = []
            conv1s = []

            loss = 0
            for k, img in enumerate(imgs_var[:-1]):
                r_img = imgs_var[k]
                n_img = imgs_var[k + 1]
                KRKiUV_cuda_T = KRKiUV_cuda_Ts[k]
                KT_cuda_T = KT_cuda_Ts[k]
                conv5, conv4, conv3, conv2, conv1 = encoder(r_img, n_img, KRKiUV_cuda_T, KT_cuda_T)
                latents.append(conv5)
                conv4s.append(conv4)
                conv3s.append(conv3)
                conv2s.append(conv2)
                conv1s.append(conv1)


            # deal with the last frame
            r_img = imgs_var[-1]
            n_img = imgs_var[-2]
            KRKiUV_cuda_T = KRKiUV_cuda_Ts[-1]
            KT_cuda_T = KT_cuda_Ts[-1]
            conv5, conv4, conv3, conv2, conv1 = encoder(r_img, n_img, KRKiUV_cuda_T, KT_cuda_T)


            latents.append(conv5)
            conv4s.append(conv4)
            conv3s.append(conv3)
            conv2s.append(conv2)
            conv1s.append(conv1)

            Y = torch.stack(latents, dim=1).cpu()

            Z = gplayer(D, Y)
            b, l, c, h, w = Y.size()
            for k, img in enumerate(imgs_var):
                conv5 = Z[:,k].view(b,c,h,w).cuda()
                conv4 = conv4s[k]
                conv3 = conv3s[k]
                conv2 = conv2s[k]
                conv1 = conv1s[k]
                pred = decoder(conv5, conv4, conv3, conv2, conv1)
                loss += compute_errors(gts_var[k], pred)

            loss = loss / len(imgs_var)

            errors.update([loss.item()])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            logger.valid_bar.update(i+1)
            if i % args.print_freq == 0:
                logger.valid_writer.write('valid: Time {} Abs Error {:.4f} ({:.4f})'.format(batch_time, errors.val[0], errors.avg[0]))
        logger.valid_bar.update(len(val_loader))

    return errors.avg, ['Val Total_loss']



if __name__ == '__main__':
    main()
