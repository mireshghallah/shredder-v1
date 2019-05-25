

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


import torch.optim as optim

import math
import csv
import numpy as np
import itertools
import matplotlib.pyplot as plt


from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import time
import torch.nn as nn
import matplotlib.pyplot as plt

import csv
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import math

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import time
import shutil
import os
import torch.nn.functional as F


# In[2]:


DATA_DIR='/home/actlab/releq/data.imagenet' #'../../../ILSVRC'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 100 # originally 256 
WORKERS = 1
PRINT_FREQ = 5
EPOCH = 40
LR = 0.001




traindir = os.path.join(DATA_DIR, 'train')
valdir = os.path.join(DATA_DIR, 'val')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))


# In[4]:


train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=(None is None),
        num_workers=WORKERS, pin_memory=True, sampler= None)

val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=WORKERS, pin_memory=True)


# # Utils

# In[5]:


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


# In[6]:


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# In[7]:


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = LR * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# In[8]:


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')





def validate(val_loader, model, model_original, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()


    model.eval()
    SNR = []
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input, target = input.to(device), target.to(device)

            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            #
            output_original = model_original(input)
            output = model.model_pt1(input)
            output = model.intermed(output)

            if i % PRINT_FREQ== 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))


        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg.item(), sum(SNR)/len(SNR)


datas=np.load("activation-4-alexnet.npy")

stds= np.std(datas,axis=0)
means= np.mean(datas,axis=0)




val_loader_2 = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=1, shuffle=False,
        num_workers=WORKERS, pin_memory=True)



model_original = models.alexnet(pretrained= True)
model_original.to(device)
criterion = nn.CrossEntropyLoss().to(device)



# In[23]:


conv_layers = []
fc_layers=[]
for i, layer in enumerate(model_original.features):
    if isinstance(layer, nn.Conv2d):
        if ((i is not 0 )):
            conv_layers.append(i)
conv_layers.append(len(model_original.features))
for i, layer in enumerate(model_original.classifier):
    if isinstance(layer, nn.Linear):
        fc_layers.append(i)
        



# # extract activation size

# In[24]:


conv_shapes=[]
for cnt2, (data, target) in enumerate(val_loader):
    for cnt,i in enumerate(conv_layers):
        data,target = data.to(device), target.to(device)

        newmodel_original =  torch.nn.Sequential(*(list(model_original.features)[0:i]))
        
        output_original = newmodel_original(data)
        conv_shapes.append(output_original.shape[1:])

    if (cnt2==0):
        break
    

print(conv_shapes)



class NoisyActivation(nn.Module):
    def __init__(self, activation_size):
        super(NoisyActivation, self).__init__()

        m =torch.distributions.laplace.Laplace(loc = torch.FloatTensor(means), scale = torch.FloatTensor(stds), validate_args=None)
        self.noise = nn.Parameter(m.sample())



  

    def forward(self, input):

        return input + self.noise

class alexnet_syn(nn.Module):

    def __init__(self, model_features, model_classifier, conv_layers, conv_shapes, index ):
        super(alexnet_syn, self).__init__()
        
        self.model_pt1 =  torch.nn.Sequential(*(list(model_features)[0:conv_layers[index]]))
        self.intermed = NoisyActivation(conv_shapes[index])
        self.model_pt2 =  torch.nn.Sequential(*(list(model_features)[conv_layers[index]:]))
        self.model_pt3 = model_classifier
        for params in self.model_pt1.parameters():
            params.requires_grad = False
        for params in self.model_pt2.parameters():
            params.requires_grad = False
        for params in self.model_pt3.parameters():
            params.requires_grad = False


    def forward(self, img):
        x = self.model_pt1(img)
        x = self.intermed (x)
        x = self.model_pt2(x)
        x = x.view(x.size(0), -1)
        x = self.model_pt3(x)

        return x
    





model_original = models.alexnet(pretrained= True)
model_original.to(device)
criterion = nn.CrossEntropyLoss().to(device)
print(model_original)



model_syn_original =torch.nn.Sequential(*(list(model_original.features)[0:conv_layers[4]]))
model_syn_original_rest = torch.nn.Sequential(*(list(model_original.classifier)))

model_syn_original.eval()
model_syn_original_rest.eval()


# In[51]:


model_syn = alexnet_syn(model_original.features, model_original.classifier, conv_layers, conv_shapes, 4)
model_syn.to(device)
criterion = nn.CrossEntropyLoss().to(device)


model_syn.eval()
total_correct = 0
avg_loss = 0.0
for i, (images, labels) in enumerate(val_loader_2):

    images , labels = images.to(device), labels.to(device)
    break


imgs =np.reshape(np.squeeze(images.cpu().detach().numpy()), (1,-1))
activation_noise = np.expand_dims( model_syn.intermed.noise.cpu().detach().numpy(),axis=0)

activation_original = np.expand_dims( model_syn.intermed.noise.cpu().detach().numpy(),axis=0)


for i, (images, labels) in enumerate(val_loader_2):
    model_original = models.alexnet(pretrained= True)
    model_original.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
 


    images, labels = images.to(device), labels.to(device)
    model_syn_original =torch.nn.Sequential(*(list(model_original.features)[0:conv_layers[4]]))
    model_syn_original_rest = torch.nn.Sequential(*(list(model_original.classifier)))

    model_syn_original.eval()
    model_syn_original_rest.eval()

    acc1 =0


    model_syn = alexnet_syn(model_original.features, model_original.classifier, conv_layers, conv_shapes, 4)
    model_syn.to(device)
    acc1, SNR = validate(val_loader, model_syn, model_syn_original, criterion)
        
    print (acc1)

    print("test and save")
    output_original = model_syn_original(images)
    output_syn = model_syn.model_pt1(images)
    output_syn= model_syn.intermed(output_syn)
 
    SNR = (((output_original**2).mean())/((model_syn.intermed.noise).var())).item()
    with open('alexnet-activation-4-laplace-MI-test.csv','a') as fd:
        writer = csv.writer(fd)
        writer.writerow([SNR, acc1])

    activation_noise=np.concatenate((activation_noise, output_syn.cpu().detach().numpy()))
    activation_original=np.concatenate((activation_original,output_original.cpu().detach().numpy()),axis=0)

    np.save("noisy-activation-4-laplace-MI-test", activation_noise)
    np.save("original-activation-4-laplace-MI-test", activation_original)

    imgs=np.concatenate((imgs,np.reshape(np.squeeze(images.detach().cpu().numpy()), (1,-1)) ))
    np.save("original-image-4-laplace-MI-test", imgs)



