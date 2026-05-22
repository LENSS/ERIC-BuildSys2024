#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 11:20:29 2022

@author: tian
"""

import pandas as pd
import matplotlib.pyplot as plt
import yaml

with open('../control.yaml') as f:
   controls = yaml.load(f, Loader=yaml.FullLoader)

input_path = controls['io_path']

data = pd.read_csv(input_path + 'accuracy.log')
data['Epoch'] = [i for i in range(1,len(data['train_loss'])+1)]

fs = (5,5)

fig, axs = plt.subplots(figsize=fs)
axs.plot(data['Epoch'], data['train_loss'],'red',label="Train", linestyle='solid',marker='o',markersize=6, alpha = 1)
axs.plot(data['Epoch'], data['val_loss'],'Blue',label="Validation", linestyle='solid',marker='^',markersize=6, alpha = 1)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
# plt.grid()
plt.savefig(input_path+'ResNet18_Loss.png')
plt.show()

fig, axs = plt.subplots(figsize=fs)
axs.plot(data['Epoch'], data['train_acc'],'red',label="Train", linestyle='solid',marker='o',markersize=6, alpha = 1)
axs.plot(data['Epoch'], data['val_acc'],'Blue',label="Validation", linestyle='solid',marker='^',markersize=6, alpha = 1)
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
# plt.grid()
plt.savefig(input_path+'ResNet18_Accuracy.png')
plt.show()
