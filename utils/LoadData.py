from .transforms import transforms
from torch.utils.data import DataLoader
import torchvision
import torch
import numpy as np
from torch.utils.data import Dataset
from .imutils import RandomResizeLong
import os
from PIL import Image
import random

def train_data_loader(args, test_path=False, segmentation=False):
    mean_vals = [0.485, 0.456, 0.406]
    std_vals = [0.229, 0.224, 0.225] 
    input_size = int(args.input_size)
    crop_size = int(args.crop_size)
    tsfm_train = transforms.Compose([transforms.Resize(input_size),  
                                     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    tsfm_test = transforms.Compose([transforms.Resize(input_size),  
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    img_train = VOCDataset(args.train_list, root_dir=args.img_dir, num_classes=args.num_classes, transform=tsfm_train, test=True)
    img_test = VOCDataset(args.test_list, root_dir=args.img_dir, num_classes=args.num_classes, transform=tsfm_test, test=True)
    train_loader = DataLoader(img_train, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader,  val_loader

def test_data_loader(args, test_path=False, segmentation=False):
    mean_vals = [0.485, 0.456, 0.406]
    std_vals = [0.229, 0.224, 0.225]
    input_size = int(args.input_size)
    tsfm_test = transforms.Compose([transforms.Resize(input_size),  
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])  

    img_test = VOCDataset(args.test_list, root_dir=args.img_dir, num_classes=args.num_classes, transform=tsfm_test, test=True)
    val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return val_loader

class VOCDataset(Dataset):
    def __init__(self, datalist_file, root_dir, num_classes=20, transform=None, test=False):
        self.root_dir = root_dir
        self.testing = test
        self.datalist_file =  datalist_file
        self.transform = transform
        self.num_classes = num_classes
        self.image_list, self.label_list = self.read_labeled_image_list(self.root_dir, self.datalist_file)
        self.image_num = 2
        self.thres = 1000

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):

        image_names = []
        images = []
        labels = []
        img_name =  self.image_list[idx]
        image = Image.open(img_name).convert('RGB')
        image = self.transform(image)
        label = self.label_list[idx]
        ind = np.where(label==1)[0]
        flag = ind
        random.shuffle(flag)
        flag = flag[0]

        for i in range(self.image_num):
            t = 0
            while True:
                idx = np.random.randint(1,len(self.image_list)-1)
                t = t + 1
                if((label == self.label_list[idx]).all()):
                    img_name_1 = self.image_list[idx]
                    image_1 = Image.open(img_name_1).convert('RGB')
                    image_1 = self.transform(image_1)
                    label_1 = self.label_list[idx]
                    image_names.append(img_name_1)
                    images.append(image_1)
                    labels.append(label_1)
                    break
                if t > self.thres:
                    if self.label_list[idx][flag] == 1:
                        img_name_1 = self.image_list[idx]
                        image_1 = Image.open(img_name_1).convert('RGB')
                        image_1 = self.transform(image_1)
                        label_1 = self.label_list[idx]
                        image_names.append(img_name_1)
                        images.append(image_1)
                        labels.append(label_1)
                        break
        
        return img_name, image, label, image_names[0], images[0], labels[0], image_names[1], images[1], labels[1]

    def read_labeled_image_list(self, data_dir, data_list):
        with open(data_list, 'r') as f:
            lines = f.readlines()
        img_name_list = []
        img_labels = []
        for line in lines:
            fields = line.strip().split()
            image = fields[0] + '.jpg'
            labels = np.zeros((self.num_classes,), dtype=np.float32)
            for i in range(len(fields)-1):
                index = int(fields[i+1])
                labels[index] = 1.
            img_name_list.append(os.path.join(data_dir, image))
            img_labels.append(labels)
        return img_name_list, img_labels
