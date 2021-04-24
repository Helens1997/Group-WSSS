import sys
import os
import random
import torch
import argparse
import numpy as np
import time
import shutil
import json
import my_optim
import torch.optim as optim
from models import vgg_gnn
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
from utils import AverageMeter
from utils.LoadData import train_data_loader
from tqdm import trange, tqdm

ROOT_DIR = '/'.join(os.getcwd().split('/')[:-1])
print('Project Root Dir:', ROOT_DIR)

def get_arguments():
    parser = argparse.ArgumentParser(description='group-wise semantic mining for weakly supervised semantic segmentation')
    parser.add_argument("--root_dir", type=str, default=ROOT_DIR)
    parser.add_argument("--img_dir", type=str, default='data/VOCdevkit/VOC2012/JPEGImages')
    parser.add_argument("--train_list", type=str, default='data/VOCdevkit/VOC2012/ImageSets/Segmentation/train_cls.txt')
    parser.add_argument("--test_list", type=str, default='data/VOCdevkit/VOC2012/ImageSets/Segmentation/val_cls.txt')
    parser.add_argument("--num_classes", type=int, default=20) 
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=1) 
    parser.add_argument("--dataset", type=str, default='pascal_voc')
    parser.add_argument("--drop_rate",type=float,default=0.8)
    parser.add_argument("--drop_th",type=float,default=0.7)
    parser.add_argument("--decay_points", type=str, default='5,10')
    parser.add_argument("--snapshot_dir", type=str, default='runs/pascal/model')
    parser.add_argument("--att_dir", type=str, default='runs/pascal/feat')
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--disp_interval", type=int, default=100)
    parser.add_argument("--resume", type=str, default='False')
    parser.add_argument("--global_counter", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=15)
    parser.add_argument("--current_epoch", type=int, default=0)

    return parser.parse_args()

def save_checkpoint(args, state, is_best, filename='checkpoint.pth.tar'):
    savepath = os.path.join(args.snapshot_dir, filename)
    torch.save(state, savepath)
    if is_best:
        shutil.copyfile(savepath, os.path.join(args.snapshot_dir, 'model_best.pth.tar'))

def get_model(args):
    model = vgg_gnn.GNNNet_VGG(pretrained=True,drop_rate=args.drop_rate,drop_th=args.drop_th, num_classes=args.num_classes,att_dir=args.att_dir, training_epoch=args.epoch)
    device = torch.device(0)	
    model = torch.nn.DataParallel(model).cuda()
    model.to(device)
    param_groups = model.module.get_parameter_groups()
    optimizer = optim.SGD([
        {'params': param_groups[0], 'lr': args.lr},
        {'params': param_groups[1], 'lr': 2*args.lr},
        {'params': param_groups[2], 'lr': 10*args.lr},
        {'params': param_groups[3], 'lr': 20*args.lr}], momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    return  model, optimizer

def train(args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    
    total_epoch = args.epoch
    global_counter = args.global_counter
    current_epoch = args.current_epoch

    train_loader, val_loader = train_data_loader(args)
    max_step = total_epoch*len(train_loader)
    args.max_step = max_step 
    print('Max step:', max_step)
    
    model, optimizer = get_model(args)
    
    model.train()
    print(model)
    end = time.time()

    while current_epoch < total_epoch:
        model.train()
        losses.reset()
        batch_time.reset()
        res = my_optim.reduce_lr(args, optimizer, current_epoch)
        steps_per_epoch = len(train_loader)

        index = 0  
        for idx, dat in enumerate(train_loader):
            
            img_name1, img1, label1, img_name2, img2, label2, img_name3, img3, label3 = dat
            label1 = label1.cuda(non_blocking=True)
            label2 = label2.cuda(non_blocking=True)
            label3 = label3.cuda(non_blocking=True)            
            
            x11, x1, x22,x2, x33,x3 = model(img1, img2, img3, current_epoch, label1, index)
            index += 1

            loss_train = (0.4 * (F.multilabel_soft_margin_loss(x11, label1) + F.multilabel_soft_margin_loss(x22, label2)
                    + F.multilabel_soft_margin_loss(x33, label3)) + (F.multilabel_soft_margin_loss(x1, label1)
                    + F.multilabel_soft_margin_loss(x2, label2) + F.multilabel_soft_margin_loss(x3, label3)) ) / 6

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            losses.update(loss_train.data.item(), img1.size()[0])
            batch_time.update(time.time() - end)
            end = time.time()
            
            global_counter += 1
            if global_counter % 1000 == 0:
                losses.reset()

            if global_counter % args.disp_interval == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'LR: {:.5f}\t' 
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                        current_epoch, global_counter%len(train_loader), len(train_loader), 
                        optimizer.param_groups[0]['lr'], loss=losses))

        if current_epoch == args.epoch-1:
            save_checkpoint(args,
                        {
                            'epoch': current_epoch,
                            'global_counter': global_counter,
                            'state_dict':model.state_dict(),
                            'optimizer':optimizer.state_dict()
                        }, is_best=False,
                        filename='%s_epoch_%d.pth' %(args.dataset, current_epoch))
        current_epoch += 1

if __name__ == '__main__':
    args = get_arguments()
    print('Running parameters:\n', args)
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    train(args)
