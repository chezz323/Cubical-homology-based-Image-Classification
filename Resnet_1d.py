import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

import torch

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv5x5(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=1, bias=False)

def conv7x7(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=1, bias=False)



class BasicBlock3x3(nn.Module):
    expansion = 1

    def __init__(self, inplanes3, planes, stride=1, downsample=None):
        super(BasicBlock3x3, self).__init__()
        self.conv1 = conv3x3(inplanes3, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock5x5(nn.Module):
    expansion = 1

    def __init__(self, inplanes5, planes, stride=1, downsample=None):
        super(BasicBlock5x5, self).__init__()
        self.conv1 = conv5x5(inplanes5, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv5x5(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        d = residual.shape[2] - out.shape[2]
        out1 = residual[:,:,0:-d] + out
        out1 = self.relu(out1)
        # out += residual

        return out1



class BasicBlock7x7(nn.Module):
    expansion = 1

    def __init__(self, inplanes7, planes, stride=1, downsample=None):
        super(BasicBlock7x7, self).__init__()
        self.conv1 = conv7x7(inplanes7, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv7x7(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        d = residual.shape[2] - out.shape[2]
        out1 = residual[:, :, 0:-d] + out
        out1 = self.relu(out1)
        # out += residual

        return out1




class MSResNet(nn.Module):
    def __init__(self, input_channel, layers=[1, 1, 1, 1], num_classes=10):
        self.inplanes3 = 64
        self.inplanes5 = 64
        self.inplanes7 = 64

        super(MSResNet, self).__init__()

        self.conv1 = nn.Conv1d(input_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer3x3_1 = self._make_layer3(BasicBlock3x3, 64, layers[0], stride=2)
        self.layer3x3_2 = self._make_layer3(BasicBlock3x3, 128, layers[1], stride=2)
        self.layer3x3_3 = self._make_layer3(BasicBlock3x3, 256, layers[2], stride=2)
        # self.layer3x3_4 = self._make_layer3(BasicBlock3x3, 512, layers[3], stride=2)

        # maxplooing kernel size: 16, 11, 6
        self.maxpool3 = nn.AvgPool1d(kernel_size=16, stride=1, padding=0)


        self.layer5x5_1 = self._make_layer5(BasicBlock5x5, 64, layers[0], stride=2)
        self.layer5x5_2 = self._make_layer5(BasicBlock5x5, 128, layers[1], stride=2)
        self.layer5x5_3 = self._make_layer5(BasicBlock5x5, 256, layers[2], stride=2)
        # self.layer5x5_4 = self._make_layer5(BasicBlock5x5, 512, layers[3], stride=2)
        self.maxpool5 = nn.AvgPool1d(kernel_size=11, stride=1, padding=0)


        self.layer7x7_1 = self._make_layer7(BasicBlock7x7, 64, layers[0], stride=2)
        self.layer7x7_2 = self._make_layer7(BasicBlock7x7, 128, layers[1], stride=2)
        self.layer7x7_3 = self._make_layer7(BasicBlock7x7, 256, layers[2], stride=2)
        # self.layer7x7_4 = self._make_layer7(BasicBlock7x7, 512, layers[3], stride=2)
        self.maxpool7 = nn.AvgPool1d(kernel_size=6, stride=1, padding=0)

        # self.drop = nn.Dropout(p=0.2)
        self.fc = nn.Linear(256*3, num_classes)

        # todo: modify the initialization
        # for m in self.modules():
        #     if isinstance(m, nn.Conv1d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm1d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_layer3(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes3 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes3, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes3, planes, stride, downsample))
        self.inplanes3 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes3, planes))

        return nn.Sequential(*layers)

    def _make_layer5(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes5 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes5, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes5, planes, stride, downsample))
        self.inplanes5 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes5, planes))

        return nn.Sequential(*layers)


    def _make_layer7(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes7 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes7, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes7, planes, stride, downsample))
        self.inplanes7 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes7, planes))

        return nn.Sequential(*layers)

    def forward(self, x0):
        x0 = self.conv1(x0)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0 = self.maxpool(x0)

        x = self.layer3x3_1(x0)
        x = self.layer3x3_2(x)
        x = self.layer3x3_3(x)
        # x = self.layer3x3_4(x)
        x = self.maxpool3(x)

        y = self.layer5x5_1(x0)
        y = self.layer5x5_2(y)
        y = self.layer5x5_3(y)
        # y = self.layer5x5_4(y)
        y = self.maxpool5(y)

        z = self.layer7x7_1(x0)
        z = self.layer7x7_2(z)
        z = self.layer7x7_3(z)
        # z = self.layer7x7_4(z)
        z = self.maxpool7(z)

        out = torch.cat([x, y, z], dim=1)

        out = out.squeeze()
        # out = self.drop(out)
        out1 = self.fc(out)

        return out1, out
    
    
import scipy.io as sio
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import time
import torch
from torch import nn
from torch.autograd import Variable

import math
import time
from tqdm.notebook import tqdm
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from collections import Counter

from Resnet_1d import *
from sklearn.metrics import confusion_matrix, classification_report, f1_score

class Exe_Model:
    def __init__(self, model):
        self.model = model
        self.repeat_acc = []
        self.repeat_epoch = []
        
        print("model loaded")
    def set_model(self, num_classes, num_epochs=100):
        self.num_classes = num_classes
        self.msresnet = self.model(input_channel=1, layers=[1, 1, 1, 1], num_classes= self.num_classes)
        self.msresnet = self.msresnet.cuda()
        self.num_epochs = num_epochs

        self.criterion = nn.CrossEntropyLoss(size_average=False).cuda()

        self.optimizer = torch.optim.Adam(self.msresnet.parameters(), lr=0.001) # original lr=0.005
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[50, 100, 150, 200, 250, 300], gamma=0.1)
        self.train_loss = np.zeros([self.num_epochs, 1])
        self.val_loss = np.zeros([self.num_epochs, 1])
        self.train_acc = np.zeros([self.num_epochs, 1])
        self.val_acc = np.zeros([self.num_epochs, 1])
        
    def train_model(self, train_data_loader, val_data_loader, num_train_instances, num_val_instances, es_tolerance=20, verbose=1):
        self.train_data_loader=train_data_loader
        self.val_data_loader=val_data_loader
        self.num_train_instances=num_train_instances
        self.num_val_instances=num_val_instances
        #self.num_train_instances
        start = time.time()
        early_stopping_counter = 0
        self.epoch_atbest = 0
        if verbose: 
            print("==============================================")
        for epoch in range(self.num_epochs):
            y_pred = []
            y_true = []
            self.msresnet.train()
            self.scheduler.step()
            # for i, (samples, labels) in enumerate(train_data_loader):
            loss_x = 0
            for (samples, labels) in (lambda x: tqdm(self.train_data_loader) if x==1 else self.train_data_loader)(verbose):
                samplesV = Variable(samples.cuda())
                labels = labels.squeeze()
                labelsV = Variable(labels.cuda())

                # Forward + Backward + Optimize
                self.optimizer.zero_grad()
                predict_label = self.msresnet(samplesV)

                loss = self.criterion(predict_label[0], labelsV)

                loss_x += loss.item()

                loss.backward()
                self.optimizer.step()

            self.train_loss[epoch] = loss_x / self.num_train_instances

            self.msresnet.eval()
            # loss_x = 0
            correct_train = 0
            for i, (samples, labels) in enumerate(self.train_data_loader):
                with torch.no_grad():
                    samplesV = Variable(samples.cuda())
                    labels = labels.squeeze()
                    labelsV = Variable(labels.cuda())
                    # labelsV = labelsV.view(-1)

                    predict_label = self.msresnet(samplesV)
                    prediction = predict_label[0].data.max(1)[1]
                    correct_train += prediction.eq(labelsV.data.long()).sum()

                    loss = self.criterion(predict_label[0], labelsV)
                    # loss_x += loss.item()
            if verbose: 
                print('Epoch:', epoch)
                print("Training accuracy:", (100*float(correct_train)/ self.num_train_instances))

            # train_loss[epoch] = loss_x / num_train_instances
            self.train_acc[epoch] = 100*float(correct_train)/ self.num_train_instances

            trainacc = str(100*float(correct_train)/self.num_train_instances)[0:6]


            loss_x = 0
            correct_val = 0
            for i, (samples, labels) in enumerate(self.val_data_loader):
                with torch.no_grad():
                    samplesV = Variable(samples.cuda())
                    labels = labels.squeeze()
                    labelsV = Variable(labels.cuda())
                    # labelsV = labelsV.view(-1)

                predict_label = self.msresnet(samplesV)
                prediction = predict_label[0].data.max(1)[1]
                correct_val += prediction.eq(labelsV.data.long()).sum()

                y_pred = np.append(y_pred, prediction.cpu())
                y_true = np.append(y_true, labels)

                loss = self.criterion(predict_label[0], labelsV)
                loss_x += loss.item()
                
            if verbose: 
                print("val accuracy:", (100 * float(correct_val) / self.num_val_instances))
                print("F1 score:", f1_score(y_true, y_pred, average='weighted'))
                print("==============================================")
            self.val_loss[epoch] = loss_x /self.num_val_instances
            self.val_acc[epoch] = 100 * float(correct_val) / self.num_val_instances

            valacc = str(100 * float(correct_val) / self.num_val_instances)[0:6]

            if epoch == 0:
                temp_val = correct_val
                temp_train = correct_train
            elif correct_val>temp_val:
                self.PATH='weights/' #train' + trainacc + 'val' + valacc + '.pkl'
                torch.save(self.msresnet, self.PATH + 'model.pt')
                torch.save(self.msresnet.state_dict(), self.PATH + 'model_state_dict.pt')
                torch.save({
                    'model': self.msresnet.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                }, self.PATH + 'all.tar')
                temp_val = correct_val
                temp_train = correct_train
                self.best_pred = y_pred
                self.best_true = y_true
                self.best_acc = float(correct_val) / self.num_val_instances
                self.f1score = f1_score(self.best_true, self.best_pred, average='weighted')
                early_stopping_counter=0
                self.epoch_atbest = epoch
                self.train_time=time.time()-start
            else:
                early_stopping_counter += 1

            if es_tolerance <= early_stopping_counter:
                #self.train_time=time.time()-start
                self.current_epoch=epoch
                print("\n Best validation accuracy:", self.best_acc, "   when epoch:", self.epoch_atbest)
                print("triain time:", self.train_time)
                break
        #self.train_time=time.time()-start
        self.current_epoch=epoch
    def repeat_train(self, n_repeat=10):
        for i in tqdm(range(n_repeat)):
            self.set_model(self.num_classes)
            self.train_model(self.train_data_loader, self.val_data_loader, self.num_train_instances, self.num_val_instances, es_tolerance=20, verbose=0)
            self.repeat_acc = np.append(self.repeat_acc, self.best_acc)
            self.repeat_epoch = np.append(self.repeat_epoch, self.epoch_atbest)
        print("Best Accuracy: %.4f" % self.repeat_acc.max(), "Mean Accuracy: %.4f" % self.repeat_acc.mean(), "Stdard deviation: %.4f" % self.repeat_acc.std(), "Mean Epochs: %.4f" % self.repeat_epoch.mean())
        
        
    def result(self):
        print(classification_report(self.best_true, self.best_pred))
        print("F1 score:", f1_score(self.best_true, self.best_pred, average='macro'))
        print("Validation Accuracy:", self.best_acc, "   when epoch:", self.epoch_atbest)
        
    def cm(self):
        cm_width = int(max(self.best_true)) +1
        result_map = confusion_matrix(self.best_true, self.best_pred)
        cm = np.zeros((cm_width,cm_width))
        for i, lst in enumerate(result_map):
            for j, k in enumerate(lst):
                cm[i][j] = lst[j]/lst.sum()
        figure = plt.figure(figsize=(cm_width,cm_width))
        sns.heatmap(cm, annot=True, cmap='crest')
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
        
    def test(self, test_data_loader, num_test_instances):
        self.test_data_loader = test_data_loader
        self.num_test_instances = num_test_instances
        y_pred = []
        y_true = []
        loss_x = 0
        correct_test = 0
        self.msresnet.eval()
        for i, (samples, labels) in enumerate(self.test_data_loader):
            with torch.no_grad():
                samplesV = Variable(samples.cuda())
                labels = labels.squeeze()
                labelsV = Variable(labels.cuda())
            

            predict_label = self.msresnet(samplesV)
            prediction = predict_label[0].data.max(1)[1]
            correct_test += prediction.eq(labelsV.data.long()).sum()

            y_pred = np.append(y_pred, prediction.cpu())
            y_true = np.append(y_true, labels)

            loss = self.criterion(predict_label[0], labelsV)
            loss_x += loss.item()
        testacc = str(100 * float(correct_test) / self.num_test_instances)[0:6]
        print("Test Accuracy:", testacc, "|  F1 score (Macro):", f1_score(y_true, y_pred, average='macro'), "|  F1 score (Weighted):", f1_score(y_true, y_pred, average='weighted'))