import argparse
import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import wandb
# used for logging to TensorBoard
from tensorboard_logger import configure, log_value

from wideresnet import WideResNet

parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
parser.add_argument('--n_subnetworks', default=3, type=int, help='number of subnetworks')
parser.add_argument('--dataset', default='cifar100', type=str,
                    help='dataset (cifar10 [default] or cifar100)')
parser.add_argument('--epochs', default=200, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--layers', default=28, type=int,
                    help='total number of layers (default: 28)')
parser.add_argument('--widen-factor', default=10, type=int,
                    help='widen factor (default: 10)')
parser.add_argument('--droprate', default=0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='WideResNet-28-10', type=str,
                    help='name of experiment')
parser.add_argument('--tensorboard',
                    help='Log progress to wandb', action='store_true')
parser.set_defaults(augment=True)

best_prec1 = 0

def C_train_collate_fn(batch, M):
    """Collate function for training MIMO on CIFAR classification"""
    
    x, y = zip(*batch)

    x, y = torch.stack(list(x)), torch.tensor(y)
    x = torch.cat(torch.chunk(x, M, dim=0), dim=1)
    y = torch.stack(torch.chunk(y, M, dim=0), dim=1)

    return x, y

def C_test_collate_fn(batch, M):
    """Collate function for testing MIMO on CIFAR classification"""
    
    x, y = zip(*batch)
    x, y = torch.stack(list(x)), torch.tensor(y)
    x = x.repeat(1, M, 1, 1)
    y = y[:,None].repeat(1,M)
    
    return x, y

def logmeanexp(x, dim=None, keepdim=False):
	to_numpy = False
	
	if not isinstance(x, torch.Tensor):
		x = torch.tensor(x)
		to_numpy = True
		
	if dim is None:
		x, dim = x.view(-1), 0
	
	x_max, _ = torch.max(x, dim, keepdim=True)
	x = x_max + torch.log(torch.mean(torch.exp(x - x_max), dim, keepdim=True))
	
	x = x if keepdim else x.squeeze(dim)
	if to_numpy:
		x = x.numpy()
		
	return x

def main():
    global args, best_prec1
    args = parser.parse_args()
    if args.tensorboard: configure("runs/%s"%(args.name))

    wandb.init(
    project="FinalRuns", 
    name=f"MIMO_WideResNet-28-10_CIFAR100_{args.n_subnetworks}_subnetworks",
    # mode='disabled',
    # name="DELETE_THIS", 
)


    # Data loading code
    # (0.5071, 0.4865, 0.4409)
    #  (0.267, 0.256, 0.276)
    # normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
    #                                  std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    normalize = transforms.Normalize(mean=(0.5071, 0.4865, 0.4409), std=(0.267, 0.256, 0.276))

    if args.augment:
        transform_train = transforms.Compose([
        	transforms.ToTensor(),
        	transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
        						(4,4,4,4),mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])

    kwargs = {'num_workers': 4, 'pin_memory': True}
    n_subnetworks = args.n_subnetworks
    assert(args.dataset == 'cifar10' or args.dataset == 'cifar100')
    train_loader = torch.utils.data.DataLoader(
        datasets.__dict__[args.dataset.upper()]('../data', train=True, download=True,
                         transform=transform_train),
        batch_size=args.batch_size*args.n_subnetworks, shuffle=True, collate_fn=lambda x: C_train_collate_fn(x, n_subnetworks), drop_last=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        datasets.__dict__[args.dataset.upper()]('../data', train=False, transform=transform_test),
        batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: C_test_collate_fn(x, n_subnetworks), **kwargs)

    # create model
    model = WideResNet(args.layers, args.dataset == 'cifar10' and 10 or 100,
                            args.widen_factor, dropRate=args.droprate, n_subnetworks=args.n_subnetworks)

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    # model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    # criterion = nn.CrossEntropyLoss().cuda()
    criterion = nn.NLLLoss(reduction='sum').cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum, nesterov = args.nesterov,
                                weight_decay=args.weight_decay)

    # cosine learning rate
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60*n_subnetworks, 120*n_subnetworks, 160*n_subnetworks], gamma=0.2)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader)*args.epochs)

    for epoch in range(args.start_epoch, args.epochs*n_subnetworks):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, scheduler, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch)
        scheduler.step()

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)
    print('Best accuracy: ', best_prec1)

def train(train_loader, model, criterion, optimizer, scheduler, epoch):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)

        # compute output
        log_probs, output, individual_outputs = model(input)
        loss = criterion(log_probs, target)

        # measure accuracy and record loss
        prec1 = accuracy(individual_outputs.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        wandb.log({"Train loss": loss.item()})
        wandb.log({"lr": scheduler.get_last_lr()[0]})
        for j in range(args.n_subnetworks):
            wandb.log({f"Train accuracy {j}": prec1[j]})
        

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      loss=losses))
            for j in range(args.n_subnetworks):
                print(f'Prec@1 {j} {top1.val[j]:.3f} ({top1.avg[j]:.3f})\t')
            
        # log to TensorBoard
        if args.tensorboard:
            log_value('train_loss', losses.avg, epoch)
            for j in range(args.n_subnetworks):
                log_value(f'train_acc {j}', top1.avg[j], epoch)


def validate(val_loader, model, criterion, epoch):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(non_blocking=True)[:,0]
        input = input.cuda(non_blocking=True)

        # compute output
        with torch.no_grad():
            log_probs, output, individual_outputs = model(input)
        log_p = logmeanexp(log_probs, dim=2)
        loss = criterion(log_p, target)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1)) 

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    wandb.log({"Val loss": losses.avg})
    wandb.log({"Val accuracy": top1.avg})

    # log to TensorBoard
    if args.tensorboard:
        log_value('val_loss', losses.avg, epoch)
        log_value('val_acc', top1.avg, epoch)

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/"%(args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/'%(args.name) + 'model_best.pth.tar')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    pred = output
    correct = pred.eq(target)

    res = []
    for k in topk:
        correct_k = correct.sum(0)
        res.append(correct_k.float().mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()