import numpy as np
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.utils.data as data
import multiprocessing
from sklearn.metrics import confusion_matrix
import sys, os
import math
# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True
import yaml
from tqdm import tqdm
import sklearn.metrics as sm
from torch.utils.data import Dataset
from skimage import io
import pandas as pd

with open('../control.yaml') as f:
   controls = yaml.load(f, Loader=yaml.FullLoader)

'''
Sample run: python eval.py
'''

class RainFallDataset2(Dataset):
   
   def __init__(self, img_dir, img_label, transform=None):
      self.img_label = pd.read_csv(img_label)
      self.img_dir = img_dir
      self.transform = transform
   
   def __getitem__(self, index):             # rain / norain folder              # img filename
      img_path = os.path.join(self.img_dir, self.img_label.iloc[index,0], self.img_label.iloc[index,1])
      image = io.imread(img_path)
      y_label = torch.tensor(float(self.img_label.iloc[index,2]))
      y_label = y_label.view(-1,)
      
      if self.transform:
         image = self.transform(image)
         
      return (image, y_label, img_path)
   
   def __len__(self):
      return len(self.img_label)

def cal_score(y_test, y_pred, mode):
   
    print(mode)
    s1= sum(y_test)
    s2= sum(y_pred)
    error_pct = abs(s1-s2)/s1
    MAE = round(sm.mean_absolute_error(y_test, y_pred), 4)
    MSE = round(sm.mean_squared_error(y_test, y_pred), 4)
    MDAE = round(sm.median_absolute_error(y_test, y_pred), 4)
    EVS = round(sm.explained_variance_score(y_test, y_pred), 4)
    R2_Score = round(sm.r2_score(y_test, y_pred), 4)

    if mode == 'val':
       fw = fw_val
    elif mode == 'test':
       fw = fw_test
       
    fw.write('%15s' % method); fw.write("\t")
    fw.write('%15.4f' % s1); fw.write("\t")
    fw.write('%15.4f' % s2); fw.write("\t")
    fw.write('%15.4f' % error_pct); fw.write("\t")
    fw.write('%15.4f' % MAE); fw.write("\t")
    fw.write('%15.4f' % MSE); fw.write("\t")
    fw.write('%15.4f' % MDAE); fw.write("\t")
    fw.write('%15.4f' % EVS); fw.write("\t")
    fw.write('%15.4f' % R2_Score)
    fw.write("\n")

    print("sum of y_test = ", s1)
    print("sum of y_pred = ", s2)
    print("error pct = ", error_pct)

    print("Mean absolute error =", MAE)
    print("Mean squared error =", MSE)
    print("Median absolute error =", MDAE)
    print("Explain variance score =", EVS)
    print("R2 score =", R2_Score)
    print('--'*20)


def run_eval(idx, data_dir):

    eval_dataset = RainFallDataset2(data_dir, rainfall_label_eval[idx], transform=eval_transform)
    
    eval_loader = data.DataLoader(eval_dataset, batch_size=bs, shuffle=False,
                                num_workers=num_cpu, pin_memory=True)

    # Enable gpu mode, if cuda available
    device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")
                 
    # Initialize the prediction and label lists
    predlist = torch.zeros(0,dtype=torch.long, device='cpu')
    lbllist = torch.zeros(0,dtype=torch.long, device='cpu')

    # Evaluate the model accuracy on the dataset
    f_pred = open(io_path + controls['eval_regress_fn'][idx], "w")
    f_pred.write('img_name,predicted_regress,label_regress\n')

    with torch.no_grad():
        for i, (images, labels, img_path) in tqdm(enumerate(eval_loader)):
            images, labels = images.to(device), labels.to(device)
            predicted = model(images)
            # _, predicted = torch.max(outputs.data, 1)
            # sample_fname, _ = eval_loader.dataset.samples[i]
            f_pred.write("{},{},{}\n".format(img_path, predicted.item(), labels.item()))

            predlist=torch.cat([predlist, predicted.view(-1).cpu()])
            lbllist=torch.cat([lbllist, labels.view(-1).cpu()])

    f_pred.close()

    # calculate scores
    if idx==0:
        mode = 'val'
    else:
        mode = 'test'
    
    cal_score(lbllist, predlist, mode)
    

# main()-----------------------------------------------------------
if __name__ == '__main__':

   io_path = controls['io_path']
   
   # Paths for image directory and model
   EVAL_DIR = [controls['val_dir'], controls['test_dir']]
   rainfall_label_eval = [controls['rainfall_label_val'], controls['rainfall_label_test']]
   
   # EVAL_MODEL='models/mobilenetv2.pth'
   EVAL_MODEL = controls['eval_regress_model']
   
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
   fw_val = open(io_path+method+'_val_rain_estimation_score.log','w')
   col_names = ['Model','sum_y_val','sum_y_pred','sum_y_pred_filtered','error_pct','MAE', 'MSE', 'MDAE','EVS','R2_Score']
   for name in col_names:
       fw_val.write('%15s' % name); fw_val.write("\t")
   fw_val.write('\n')
   
   fw_test = open(io_path+method+'_test_rain_estimation_score.log','w')
   col_names = ['Model','sum_y_test','sum_y_pred','sum_y_pred_filtered','error_pct','MAE', 'MSE', 'MDAE','EVS','R2_Score']
   for name in col_names:
       fw_test.write('%15s' % name); fw_test.write("\t")
   fw_test.write('\n')
   
   # Prepare the eval data loader
   eval_transform=transforms.Compose([
               transforms.ToPILImage(),
               transforms.Resize(size=224),
               #transforms.RandomResizedCrop(size=256, scale=(0, 0.8)), # crop and then resize
               #transforms.CenterCrop(size=224),
               transforms.ToTensor(),
               transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
               ])
   
   for i, data_dir in enumerate(EVAL_DIR):
       run_eval(i, data_dir)
   
   fw_val.close()
   fw_test.close()