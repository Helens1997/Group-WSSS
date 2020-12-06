import numpy as np
import cv2
import logging
import os 
from os.path import exists 
import imageio
import torch
import matplotlib.pyplot as plt

    
def load_dataset(test_lst):
    logging.info('Beginning loading dataset...')
    im_lst = []
    label_lst = []
    with open(test_lst) as f:
        test_names = f.readlines()
    lines = open(test_lst).read().splitlines()
    for line in lines:
        fields = line.split()
        im_name = fields[0]
        im_labels = []
        for i in range(len(fields)-1):
            im_labels.append(int(fields[i+1]))
        im_lst.append(im_name)
        label_lst.append(im_labels)
    return im_lst, label_lst

if __name__ == '__main__':
    
    train_lst = 'data/VOCdevkit/VOC2012/ImageSets/Segmentation/train_cls.txt'
    root_folder = 'data/VOCdevkit/VOC2012'
    im_lst, label_lst = load_dataset(train_lst)
    
    atten_path = 'feat_1'
    save_path = 'orig_1'
    if not exists(save_path):
        os.mkdir(save_path)
    for i in range(len(im_lst)):
        im_name = '{}/JPEGImages/{}.jpg'.format(root_folder, im_lst[i])
        im_labels = label_lst[i]
        
        img = cv2.imread(im_name)
        height, width = img.shape[:2]
        
        for label in im_labels:
            att_name = '{}/{}_{}.png'.format(atten_path, i, label)
            if not exists(att_name):
                continue 
            att = cv2.imread(att_name, 0)
            att = cv2.resize(att, (width, height))
            att = np.array(att, dtype = np.float32)
            save_name = '{}/{}_{}.png'.format(save_path, im_lst[i], label)
 
            cv2.imwrite(save_name,att)

