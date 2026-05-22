import numpy as np
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.utils.data as data
import multiprocessing
from sklearn.metrics import confusion_matrix
import sys
import math
# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True
import yaml
from tqdm import tqdm
import time

with open('../control.yaml') as f:
   controls = yaml.load(f, Loader=yaml.FullLoader)

'''
Sample run: python eval.py data/rgb/eval/
'''

def cal_score(cm, method, mode):
   
    print(mode)
    print(cm)

    TP=cm[1][1]
    TN=cm[0][0]
    FP=cm[0][1]
    FN=cm[1][0]
    '''
    print("TP = ", TP)
    print("TN = ", TN)
    print("FP = ", FP)
    print("FN = ", FN)
    '''
    acc=(TP+TN)/(TP+TN+FP+FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1=2*TP/(2*TP+FP+FN)
    MCC=(TP*TN-FP*FN)/(math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))

    print("accuracy = ", acc)
    print("precision = ", precision)
    print("recall = " , recall)
    print("F1 = ", F1)
    print("MCC = ", MCC)

    print("--"*20)
    
    if mode == 'val':
       fw = fw_val
    elif mode == 'test':
       fw = fw_test
    
    fw.write('%15s' % method); fw.write("\t")
    fw.write('%15d' % TP); fw.write("\t")
    fw.write('%15d' % TN); fw.write("\t")
    fw.write('%15d' % FP); fw.write("\t")
    fw.write('%15d' % FN); fw.write("\t")
    fw.write('%15.4f' % acc); fw.write("\t")
    fw.write('%15.4f' % precision); fw.write("\t")
    fw.write('%15.4f' % recall); fw.write("\t")
    fw.write('%15.4f' % F1); fw.write("\t")
    fw.write('%15.4f' % MCC); 
    fw.write("\n")

def run_eval(num, data_dir):

    eval_dataset=datasets.ImageFolder(root=data_dir, transform=eval_transform)

    eval_loader=data.DataLoader(eval_dataset, batch_size=bs, shuffle=False,
                                num_workers=num_cpu, pin_memory=True)

    # Enable gpu mode, if cuda available
    device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")

    # Number of classes and dataset-size
    num_classes = len(eval_dataset.classes)
    print('\neval_dataset classes =', eval_dataset.classes)
    dsize=len(eval_dataset)

    # Class label names
    # class_names=['norain','rain']
                 
    # Initialize the prediction and label lists
    predlist = torch.zeros(0,dtype=torch.long, device='cpu')
    lbllist = torch.zeros(0,dtype=torch.long, device='cpu')

    # Evaluate the model accuracy on the dataset
    f_pred = open(io_path + controls['eval_fn'][num], "w")
    f_pred.write('img_name,predicted,label\n')

    correct = 0
    total = 0
    # with torch.no_grad():
    #     for images, labels in eval_loader:
    #         images, labels = images.to(device), labels.to(device)
    #         outputs = model(images)
    #         _, predicted = torch.max(outputs.data, 1)

    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()

    #         predlist=torch.cat([predlist,predicted.view(-1).cpu()])
    #         lbllist=torch.cat([lbllist,labels.view(-1).cpu()])

    img_names = []
    pred_lst = []
    label_lst = []

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(eval_loader, 0)):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            sample_fname, _ = eval_loader.dataset.samples[i]

            img_names.append(sample_fname)
            pred_lst.append(predicted.item())
            label_lst.append(labels.item())

            # f_pred.write("{},{},{}\n".format(sample_fname, predicted.item(), labels.item()))

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            predlist=torch.cat([predlist,predicted.view(-1).cpu()])
            lbllist=torch.cat([lbllist,labels.view(-1).cpu()])

    # Overall accuracy
    overall_accuracy=100 * correct / total
    print("--------No temporal filtering")
    print('Raw accuracy (no temporal filtering) of the network on the {:d} test images: {:.2f}%'.format(dsize, overall_accuracy))

    # Confusion matrix
    conf_mat=confusion_matrix(lbllist.numpy(), predlist.numpy())
    print('Confusion Matrix')
    print('-'*16)
    print(conf_mat,'\n')

    print("--------with temporal filtering")
    # add temporal filtering, on a single img basis +++++
    pred_lst_filter = []
    pred_lst_filter.append(pred_lst[0])
    for i in range(1,len(pred_lst)-1):
        if pred_lst[i] == 1 and (pred_lst[i-1] ==1 or pred_lst[i+1] ==1):
            pred_lst_filter.append(1)
        else:
            pred_lst_filter.append(0)
    pred_lst_filter.append(pred_lst[-1])

    conf_mat=confusion_matrix(lbllist.numpy(), pred_lst_filter)
    print('Confusion Matrix')
    print('-'*16)
    print(conf_mat,'\n')

    # write out filtered detection results
    for i in range(len(pred_lst_filter)):
        f_pred.write("{},{},{}\n".format(img_names[i], pred_lst_filter[i], label_lst[i]))
    f_pred.close()

    # calculate scores
    if num==0:
        mode = 'val'
    else:
        mode = 'test'

    cal_score(conf_mat, method, mode)

    # Per-class accuracy
    class_accuracy=100*conf_mat.diagonal()/conf_mat.sum(1)
    print('Per class accuracy')
    print('-'*18)
    for label, accuracy in zip(eval_dataset.classes, class_accuracy):
        #  class_name=class_names[int(label)]
         class_name = label
         print('Accuracy of class %8s : %0.2f %%'%(class_name, accuracy))


#------------------------------------- main

t_s = time.time()

io_path = controls['io_path']

# Paths for image directory and model
EVAL_DIR = [controls['val_dir'], controls['test_dir']]

# EVAL_DIR = [controls['val_dir']]

# EVAL_MODEL='models/mobilenetv2.pth'
EVAL_MODEL = controls['eval_model']

# Load the model for evaluation
model = torch.load(EVAL_MODEL)
model.eval()

# Configure batch size and nuber of cpu's
# num_cpu = multiprocessing.cpu_count()
num_cpu = controls['num_cpu']
gpu_id = controls['gpu']
method = controls['model']

bs = 1

# log file
fw_val = open(io_path+method+'_val_rain_detection_score.tsv','w')
col_names = ['Model','TP','TN','FP','FN', 'Accuracy', 'Precision','Recall',
             'F1 Score','MCC']
for name in col_names:
    fw_val.write('%15s' % name); fw_val.write("\t")
fw_val.write('\n')

fw_test = open(io_path+method+'_test_rain_detection_score.tsv','w')
col_names = ['Model','TP','TN','FP','FN', 'Accuracy', 'Precision','Recall',
             'F1 Score','MCC']
for name in col_names:
    fw_test.write('%15s' % name); fw_test.write("\t")
fw_test.write('\n')


# Prepare the eval data loader
eval_transform=transforms.Compose([
        transforms.Resize(size=224),
        # transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])

for i, data_dir in enumerate(EVAL_DIR):
    run_eval(i, data_dir)

fw_val.close()
fw_test.close()

t_e = time.time()

print("=="*10, 'Time = {:.4f} s'.format(t_e-t_s))

