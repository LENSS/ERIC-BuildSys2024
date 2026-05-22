import numpy as np
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from nets import *
import time, os, copy, argparse
import multiprocessing
from torchsummary import summary
from matplotlib import pyplot as plt
import time
from tqdm import tqdm
import yaml

with open('../control.yaml') as f:
   controls = yaml.load(f, Loader=yaml.FullLoader)

#from PIL import ImageFile
#ImageFile.LOAD_TRUNCATED_IMAGES = True

'''
Sample run: python train.py --mode=finetune --model=resnet18
'''

def train_model(device, model, criterion, optimizer, scheduler, num_epochs=30):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1000
    best_epoch = 1

    f_acc = open(output_path+'accuracy.log', 'a')
    f_acc.write('epoch,train_loss,train_acc,val_loss,val_acc\n')

    # Tensorboard summary
    writer = SummaryWriter()
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        start = time.time()        

        # Each epoch has a training and validation phase
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0

        for phase in ['train', 'valid']:
            if phase == 'train':
                # print('check 1')
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # print('check 2')
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        # print('check 3')
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                # print('check 4')
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = float(running_corrects) / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'train':
                train_loss = epoch_loss
                train_acc = epoch_acc

            if phase == 'valid':
                val_loss = epoch_loss
                val_acc = epoch_acc

            # Record training loss and accuracy for each phase
            if phase == 'train':
                writer.add_scalar('Train/Loss', epoch_loss, epoch)
                writer.add_scalar('Train/Accuracy', epoch_acc, epoch)
                writer.flush()
            else:
                writer.add_scalar('Valid/Loss', epoch_loss, epoch)
                writer.add_scalar('Valid/Accuracy', epoch_acc, epoch)
                writer.flush()

        # deep copy the model
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch+1
            best_model_wts = copy.deepcopy(model.state_dict())

        end = time.time()

        print("Time takes = {} s".format(end-start))
        print()
        f_acc.write('{},{},{},{},{}\n'.format(epoch+1,train_loss,train_acc,val_loss,val_acc))
        f_acc.flush()

        # Save the checkpoint
        if (epoch+1)%save_freq == 0:
            print("\nSaving the checkpoint model...\n")
            torch.save(model, (output_path+'model_epoch_'+str(epoch+1)))
    
    f_acc.close()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f} in epoch {}'.format(best_loss, best_epoch))

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model


if __name__ == '__main__':

    # Set training mode
    train_mode = controls["mode"]
    model = controls["model"]

    print('train_mode =', train_mode)
    print('model =',model)
    
    # Set the train and validation directory paths
    train_directory = controls['train_dir']
    valid_directory = controls['val_dir']
    
    # output path
    output_path = controls['io_path']
    gpu_id = controls['gpu']

    if not os.path.isdir(output_path):
        os.mkdir(output_path)
        print("Folder {} created".format(output_path))
    else:
        print("Folder {} already existed".format(output_path))
    
    # Batch size
    bs = controls['bs']
    # Number of epochs
    num_epochs = controls['epoch']
    
    # frequency to save model checkpoint
    global save_freq
    save_freq = controls['save_fq']

    # Number of classes
    num_classes = 2
    
    # Number of workers
    num_cpu = multiprocessing.cpu_count()
    print('num_cpu =', num_cpu)
    num_cpu = controls['num_cpu']
    
    # Applying transforms to the data
    image_transforms = { 
        'train': transforms.Compose([
            transforms.Resize(size=224),
            #transforms.RandomResizedCrop(size=256, scale=(0, 0.8)), # crop and then resize
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(),
            #transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(size=224),
            #transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    }
     
    # Load data from folders
    dataset = {
        'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
        'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid'])
    }
     
    # Size of train and validation data
    global dataset_sizes
    dataset_sizes = {
        'train':len(dataset['train']),
        'valid':len(dataset['valid'])
    }
    
    # Create iterators for data loading
    dataloaders = {
        'train':data.DataLoader(dataset['train'], batch_size=bs, shuffle=True,
                                num_workers=num_cpu, pin_memory=True, drop_last=False),
        'valid':data.DataLoader(dataset['valid'], batch_size=bs, shuffle=False,
                                num_workers=num_cpu, pin_memory=True, drop_last=False)
    }
    
    # Class names or target labels
    class_names = dataset['train'].classes
    print("Classes:", class_names)
     
    # Print the train and validation data sizes
    print("Training-set size:",dataset_sizes['train'],
          "\nValidation-set size:", dataset_sizes['valid'])
    
    # Set default device as gpu, if available
    device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")
    
    if model == 'resnet18':

        if train_mode=='finetune':
         
            # Load a pretrained model - Resnet18
            print("\nLoading resnet18 for finetuning ...\n")
            model_ft = models.resnet18(pretrained=True)
        
            # Modify fc layers to match num_classes
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
        
        elif train_mode=='scratch':
            print("\nLoading resnet18 for training from scratch ...\n")
            model_ft = models.resnet18(pretrained=False)
        
            # Modify fc layers to match num_classes
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)        
            
            
            # Load a custom model - VGG11
            # print("\nLoading VGG11 for training from scratch ...\n")
            # model_ft = MyVGG11(in_ch=3,num_classes=11)
        
            # # Set number of epochs to a higher value
            # num_epochs=100
        
        elif train_mode=='transfer':
            # Load a pretrained model - MobilenetV2
            print("\nLoading mobilenetv2 as feature extractor ...\n")
            model_ft = models.mobilenet_v2(pretrained=True)    
        
            # Freeze all the required layers (i.e except last conv block and fc layers)
            for params in list(model_ft.parameters())[0:-5]:
                params.requires_grad = False
        
            # Modify fc layers to match num_classes
            num_ftrs=model_ft.classifier[-1].in_features
            model_ft.classifier=nn.Sequential(
                nn.Dropout(p=0.2, inplace=False),
                nn.Linear(in_features=num_ftrs, out_features=num_classes, bias=True)
                )    
    
    # Transfer the model to GPU
    model_ft = model_ft.to(device)
    
    # Print model summary
    print('Model Summary:-\n')
    # for num, (name, param) in enumerate(model_ft.named_parameters()):
    #     print(num, name, param.requires_grad )

    summary(model_ft, input_size=(3, 224, 224))
    #print(model_ft)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer 
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=controls['lr'], momentum=0.9)
    
    # Learning rate decay
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    
    # Model training routine 
    print("\nTraining:-\n")
    
    # Train the model
    model_ft = train_model(device, model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)
    
    # Save the entire model
    print("\nSaving the best model...")
    torch.save(model_ft, output_path+'/rain_detector_model.best')

