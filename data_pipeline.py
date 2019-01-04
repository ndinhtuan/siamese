from __future__ import print_function 
import torch 
import cv2 
import numpy 
import os 
from torch.utils.data import Dataset, DataLoader 
from torchvision import transforms 
import random

top_dir = "orl_faces"
subject_names = [i for i in os.listdir(top_dir) if os.path.isdir(top_dir+"/"+i)]

def load_imgs():

    subject_imgs = [None]*len(subject_names) #list consist of all face data of each subject

    for idx, i in enumerate(subject_names):
        imgs = [] 
        for j in os.listdir(top_dir+'/'+i):
            imgs.append(cv2.resize(cv2.imread(top_dir+'/'+i+'/'+j, -1), (46, 56)))

        subject_imgs[idx] = imgs

    return subject_imgs 

# Split for train and test
def split_subjects(subject_imgs, num_trains=35):

    return subject_imgs[0:num_trains], subject_imgs[num_trains:]

class AddressImg:

    def __init__(self, subject_idx, img_idx):
        self.subject_idx = subject_idx 
        self.img_idx = img_idx 

def create_genuine_pairs(subject_imgs):
    
    genuine_pairs = []
    labels = None 

    for subject_idx, imgs in enumerate(subject_imgs):
        for img_idx1 in range(len(imgs)):
            for img_idx2 in range(len(imgs)):
                tmp = [AddressImg(subject_idx, img_idx1), AddressImg(subject_idx, img_idx2)]
                genuine_pairs.append(tmp)

    labels = [0]*len(genuine_pairs)
    return genuine_pairs, labels

def create_impostor_pairs(subject_imgs):
    impostor_pairs = [] 
    labels = [] 

    for subject_idx1, imgs1 in enumerate(subject_imgs):
        for subject_idx2, imgs2 in enumerate(subject_imgs):

            if subject_idx1 < subject_idx2:
                for img_idx1 in range(len(imgs1)):
                    for img_idx2 in range(len(imgs2)):
                        tmp = [AddressImg(subject_idx1, img_idx1), AddressImg(subject_idx2, img_idx2)]
                        impostor_pairs.append(tmp)
    labels = [1]*len(impostor_pairs)
    return impostor_pairs, labels 

class SiameseData(Dataset):

    def __init__(self, subject_imgs, list_pairs, list_labels, transform=transforms.ToTensor()):
        
        self.list_pairs = list_pairs 
        self.subject_imgs = subject_imgs 
        self.list_labels = list_labels 
        self.transform = transform 

    def __len__(self):
        
        return len(self.list_pairs)

    def __getitem__(self, idx):

        label = self.list_labels[idx]
        add_img1, add_img2 = self.list_pairs[idx] 
        img1 = self.subject_imgs[add_img1.subject_idx][add_img1.img_idx]
        img2 = self.subject_imgs[add_img2.subject_idx][add_img2.img_idx]

        if self.transform is not None :
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return (img1, img2), label 

def shuffle_data(genuine_pairs, gen_labels, impostor_pairs, impos_labels):
    
    all_data = [] 

    for pair, label in zip(genuine_pairs, gen_labels):
        all_data.append((pair, label))

    for pair, label in zip(impostor_pairs, impos_labels):
        all_data.append((pair, label))

    random.shuffle(all_data)
    pair_datas = [i[0] for i in all_data]
    labels = [i[1] for i in all_data]

    return pair_datas, labels 

if __name__ == "__main__":
    subject_imgs = load_imgs()
    print(len(subject_imgs))
    print(len(subject_imgs[0]))
    train, test = split_subjects(subject_imgs, 35)
    print(len(train))
    print(train[0][0].shape)
    cv2.imshow("img", train[0][0])
    cv2.waitKey(0)

    #create train pairs
    genuine_pairs, gen_labels = create_genuine_pairs(train)
    print(len(genuine_pairs))

    impostor_pairs, impos_labels = create_impostor_pairs(train)
    print(len(impostor_pairs))
    pair_datas, labels = shuffle_data(genuine_pairs, gen_labels, impostor_pairs, impos_labels)
    print(len(pair_datas))

    dataset = SiameseData(subject_imgs, pair_datas, labels, None)
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=1)
    print(len(dataloader))

    for (i1, i2), l in dataset:
        pass
        #print(i1.shape)
