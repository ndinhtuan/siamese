from data_pipeline import load_imgs, split_subjects, create_genuine_pairs, create_impostor_pairs, shuffle_data, SiameseData 
from torch.utils.data import DataLoader 
from model import Siamese, SiameseLoss 
import torch 
import numpy as np
import random 

def shuffle(datas, labels):
    
    combine = []
    for data, l in zip(datas, labels):
        combine.append((data, l))
    random.shuffle(combine)
    return [i[0] for i in combine], [i[1] for i in combine]

def evaluate(dataset, model):
    right = 0
    num = 0

    for i, ((i1, i2), l) in enumerate(dataset):
        pred = model(i1, i2)
        num += l.size(0)
        right += (sum(np.argmax(pred.detach().numpy(), axis=1) == l).type(torch.FloatTensor)/l.size(0)).item()
        print "Acc test: {}".format(right*1.0/num)

    print "Acc test: {}".format(right*1.0/num)

from aug_data import soft_aug 
if __name__ == "__main__":
    subject_imgs = load_imgs()
    train, test = split_subjects(subject_imgs, 35)

    #create train pairs
    genuine_pairs, gen_labels = create_genuine_pairs(train)
    gen_pairs_test, gen_labels_test = create_genuine_pairs(test)

    impostor_pairs, impos_labels = create_impostor_pairs(train)
    impo_pairs_test, impo_labels_test = create_impostor_pairs(test)
    impostor_pairs, impos_labels = shuffle(impostor_pairs, impos_labels)

    pair_datas, labels = shuffle_data(genuine_pairs, gen_labels, impostor_pairs[:5000], impos_labels[:5000])
    pair_data_test, labels_test = shuffle_data(gen_pairs_test, gen_labels_test, impo_pairs_test, impo_labels_test)

    dataset = SiameseData(subject_imgs, pair_datas, labels, None)
    dataset_test = SiameseData(subject_imgs, pair_data_test, labels_test, None)
    dataloader = DataLoader(dataset, batch_size=32)
    dataloader_test = DataLoader(dataset_test, batch_size=32)
    
    model = Siamese()
    #loss = SiameseLoss() 
    loss = torch.nn.CrossEntropyLoss()
    #optim = torch.optim.Adam(model.parameters())
    optim = torch.optim.SGD(model.parameters(), lr = 0.001, momentum=0.9)

    for i, ((i1, i2), l) in enumerate(dataloader):

        pred = model(i1, i2)
        print(np.argmax(pred.detach().numpy(), axis=1), l)
        print("Acc: ", sum(np.argmax(pred.detach().numpy(), axis=1) == l).type(torch.FloatTensor)/l.size(0))
        _loss = loss(pred, l)
        print _loss.item()
        optim.zero_grad()
        _loss.backward()
        optim.step() 

        if i % 25 == 0:
            evaluate(dataloader_test, model)

