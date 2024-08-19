import torch.nn.functional
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import rasterio
from torchvision.transforms import v2 
import pandas as pd
import numpy as np
import os
from torchvision.models import resnet50, efficientnet_v2_s, resnet101
from torch import nn, Generator
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts
import torch
import seaborn
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from imblearn.over_sampling import SMOTE
from torch.utils.tensorboard import SummaryWriter

# ds = gdal.Open("dso_internship/2020-01-01-4596500.tif")
# myarray = np.array(ds.GetRasterBand(1).ReadAsArray())
# plt.imshow(myarray)
# plt.show()

# src = rasterio.open("dso_internship/2020-02-05-443378.tif")
# image = src.read(1)
# plt.imshow(image)
# plt.show()

def groupLabels(VesselType):
    if math.isnan(VesselType) or VesselType <= 19:
        return VesselType
    else:
        firstdigit = int(VesselType/10)
        if firstdigit not in [2,4,6,7,8,9]:
            return VesselType
        return firstdigit*10

def calculateClassWeights(VesselType):
    weights = []
    totalcount = len(VesselType)
    valuecounts = VesselType.value_counts().sort_index()
    for value, count in valuecounts.items():
        print(value, count)
        weight = totalcount/(len(valuecounts)*count)
        weights.append(weight)
    return torch.FloatTensor(weights)

def getMeanandStd(df, imagefolder):
    pixels = []
    for path in df['chipname']:
        src = rasterio.open(os.path.join(imagefolder,path))
        image = src.read()
        pixels.extend(image.flatten())
    mean = np.nanmean(pixels)
    std = np.nanstd(pixels)
    print(mean, std)
    return [mean], [std]
    #return mean and std for all images

def getOptimizer(model, optimizername, learningrate):
    if optimizername == 'AdamW':
        return AdamW(model.parameters(), lr = learningrate, weight_decay=weight_decay)
    elif optimizername == 'Adam':
        return Adam(model.parameters(), lr = learningrate, weight_decay=weight_decay)
    else:
        return SGD(model.parameters(), lr = learningrate, weight_decay=weight_decay)

def getOversampler(dataset):
    labels = dataset.labels
    class_sample_count = np.array([len(np.where(labels == t)[0]) for t in np.unique(labels)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in labels])
    # print(samples_weight)
    sampler = WeightedRandomSampler(weights=samples_weight, num_samples= len(samples_weight))
    return sampler

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = (self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss).mean()
        return loss
    
# class ReplaceNaN(object): #Function to replace NaN values in image
#     def __call__(self, tensor):
#         nan_mask = torch.isnan(tensor)
#         not_nan = tensor[~nan_mask]
#         tensor[nan_mask] = torch.median(not_nan)
#         return tensor.float()

class RandomDataAugment(object):
    def __init__(self, degrees, chance):
        self.degrees = degrees
        self.chance = chance
    def __call__(self, image):
        # Calculate median value for the current image
        median_value = float(np.nanmedian(image))
        
        #Offset range
        max_offset = 0.1
        offset = (max_offset, max_offset)
        
        #Random horizontal and vertical flip
        flip_transform = v2.Compose([
            v2.RandomHorizontalFlip(p = self.chance),
            v2.RandomVerticalFlip(p = self.chance)
        ])
        image_flipped = flip_transform(image)
        
        # Apply affine transformation with median value as fill
        affine_transform = v2.RandomAffine(degrees=self.degrees, fill=median_value, translate= offset)
        transformed_image = affine_transform(torch.tensor(image_flipped))
        
        #Apply gaussian blur
        # blur_transform = v2.GaussianBlur(3)
        # transformed_image = blur_transform(transformed_image)    
            
        #Apply contrast and brightness change
        # colour_transform = v2.ColorJitter(brightness=0.2, contrast=0.2)
        # transformed_image = colour_transform(transformed_image)
        # plt.imshow(torch.squeeze(transformed_image).numpy())
        # plt.show()
        return transformed_image

class SMOTEDataset(Dataset):
    def __init__(self, df, imagefolder, transform, augment):
        self.df = df.reset_index()
        self.transform = transform
        self.imagefolder = imagefolder
        self.augment = augment
        self.images, self.labels = self.read_smote_images()
        
    def read_smote_images(self):
        images = []
        labels = []
        for index in self.df.index:
            #read, fill NaN and normalise the images
            src = rasterio.open(os.path.join(self.imagefolder,self.df.loc[index, 'chipname']))
            image = src.read()
            median = np.nanmedian(image)
            image[np.isnan(image)] = median
            image = np.transpose(image, (1, 2, 0))
            transformed_image = self.transform(image)
            #for dual channels padding of third channel with zero tensor
            thirdband = torch.zeros((1,transformed_image.shape[1], transformed_image.shape[2]))
            transformed_image = torch.cat((transformed_image, thirdband), 0)
            images.append(transformed_image.float())
            label = self.df.loc[index, "VesselType"]
            labels.append(label)
        #perform smote
        images = np.stack(images)
        labels = np.array(labels)
        smote = SMOTE(random_state= 22)
        images_reshaped = images.reshape(images.shape[0], images.shape[1] * cropsize * cropsize) #flattens each image into a row for SMOTE,(N, C, H, W) to (N,C*H*W)
        images_resampled, labels_resampled = smote.fit_resample(images_reshaped, labels) #resample with SMOTE
        images_resampled = np.reshape(images_resampled,[images_resampled.shape[0], images.shape[1], cropsize, cropsize]) #unflatten images into (N, C, H, W)
        return images_resampled, labels_resampled  
     
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        if self.augment:
            augmentation = RandomDataAugment(degrees=180, chance=0.5)
            image = augmentation(image)       
        return image, label

class GeotiffDataset(Dataset):
    def __init__(self, df, imagefolder, transform, augment):
        self.df = df.reset_index()
        self.transform = transform
        self.imagefolder = imagefolder
        self.augment = augment
        self.images, self.labels = self.read_images()

    def read_images(self):
        #read, fill NaN and normalise the images
        images = []
        labels = []
        for index in self.df.index:
            src = rasterio.open(os.path.join(self.imagefolder,self.df.loc[index, 'chipname']))
            image = src.read() 
            median = np.nanmedian(image)
            image[np.isnan(image)] = median
            image = np.transpose(image, (1, 2, 0)) #reshape array into H,W,C to load into tensor
            transformed_image = self.transform(image)
            #for dual channels padding of third channel with zero tensor
            thirdband = torch.zeros((1,transformed_image.shape[1], transformed_image.shape[2])) 
            transformed_image = torch.cat((transformed_image, thirdband), 0)
            images.append(transformed_image.float())
            label = self.df.loc[index, "VesselType"]
            labels.append(label)
        return images, labels

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        if self.augment:
            augmentation = RandomDataAugment(degrees=5, chance=0.5)
            image = augmentation(image)
        return image, label

if __name__ == '__main__':
    datasetfolder = '12kdatasetdualchannel'
    csvfile = os.path.join(datasetfolder, 'availablechips.csv')  
    imagefolder = 'dso_internship'
    modelname = 'resnet'
    batchsize = 100
    optimizername = 'SGD'
    epochruns = 300
    learningrate = 1e-4
    balance_technique = 'oversampler' # SMOTE, weighted or oversampler
    use_scheduler = True
    t_max_value = 50
    lr_patience = 1 #actual patience is multiplied by 10 
    lr_reduction_factor = 0.1
    schedulername = f'cosineannealingwarmrestarttmax{t_max_value}' 
    label_smoothing = 0.1
    grad_clip = False
    weight_decay = 1e-4
    use_augment = True
    cropsize = 128
    criteria = 100 #minimum number of label counts to be included
    title = f"c{modelname}101p_{optimizername}_epoch{epochruns}_lr{learningrate}_{balance_technique}_{schedulername if use_scheduler else ''}labelsmooth{label_smoothing}_weightdecay{weight_decay}_{'gradclipped_' if grad_clip else ''}11kdatasetnormed_{'augmentedaggressive_' if use_augment else ''}"

    #read csv
    df = pd.read_csv(csvfile, index_col= [0])
    #apply group labels
    df['VesselType'] = df['VesselType'].apply(groupLabels)
    #remove labels with 90('Other' ship type) as VesselType
    df = df.drop(index = df[df['VesselType'] == 90].index)
    #eliminate labels with less than 100 counts
    counts = df['VesselType'].value_counts()
    df = df[df['VesselType'].isin(counts[counts >= criteria].index)]
    #save remaining labels into dictionary for mapping to predicted labels
    target_labels = dict()
    for index, label in enumerate(df['VesselType'].unique()):
        target_labels[label] = int(index)
    #note this overwrites the 'VesselType' column with the classification label
    df['VesselType'] = df['VesselType'].map(target_labels)

    #load data into dataloaders
    train, test = train_test_split(df, train_size=0.8, test_size=0.2, random_state=22)
    
    # save for gradcam
    # note this saves the 'VesselType' column with the classification label
    train.to_csv(os.path.join(datasetfolder, 'train.csv'), index= False)
    test.to_csv(os.path.join(datasetfolder, 'test.csv'), index= False)

    train = pd.read_csv(os.path.join(datasetfolder, 'train.csv'), index_col= [0])
    test = pd.read_csv(os.path.join(datasetfolder, 'test.csv'), index_col= [0])

    train_mean, train_std = getMeanandStd(train, imagefolder)
    transform_list = [

            v2.Normalize(mean = train_mean, std=train_std),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.CenterCrop(cropsize), #CenterCrop image to cropsize
            # v2.Lambda(lambda x: x.repeat(3,1,1))#for single channel uncomment
        ]

    transform = v2.Compose(transform_list)
    if balance_technique == 'SMOTE':
        traindataset = SMOTEDataset(train, imagefolder= imagefolder, transform= transform, augment = use_augment)
    else:
        traindataset = GeotiffDataset(train, imagefolder= imagefolder, transform= transform, augment = use_augment)

    if balance_technique == 'oversampler':
        oversampler = getOversampler(traindataset)
        trainloader = DataLoader(traindataset, batch_size= batchsize, sampler=oversampler)
    else:
        trainloader = DataLoader(traindataset, batch_size= batchsize, shuffle= True)

    testdataset = GeotiffDataset(test, imagefolder= imagefolder, transform= transform, augment = False)
    testloader = DataLoader(testdataset, batch_size= batchsize)

    #set device to train model
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    if modelname == 'resnet':
        model = resnet101(weights = "DEFAULT")
        # model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) #adjust the input layer to take 1 channel
        num_ftrs = model.fc.in_features #obtain number of input features in the fully connected layer fc
        model.fc = nn.Linear(num_ftrs, len(target_labels)) #change output labels for the fully connected layer (last layer)
    elif modelname == 'efficientnet':
        model = efficientnet_v2_s()
        # model.features[0][0] = nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False) 
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, len(target_labels))   
    else:
        print('Error incorrect model')
        exit()
    weights = calculateClassWeights(df['VesselType']).to(device)
    if balance_technique == 'SMOTE' or balance_technique == 'oversampler':
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(device)
    else: #for manual class weights
        criterion = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing = label_smoothing).to(device)

    optimizer = getOptimizer(model, optimizername, learningrate) #fit optimizer to resnet model
    if use_scheduler:
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=t_max_value)
    model.to(device)

    predicted_labels = []
    actual_labels = []
    modelfolder = os.path.join(datasetfolder, 'models')
    if not os.path.exists(modelfolder):
        os.mkdir(modelfolder)
    confusionmatrixfolder = os.path.join(datasetfolder, 'confusionmatrix')
    if not os.path.exists(confusionmatrixfolder):
        os.mkdir(confusionmatrixfolder)
    logfolder = os.path.join(datasetfolder, 'logs')
    if not os.path.exists(logfolder):
        os.mkdir(logfolder)
    writer = SummaryWriter(log_dir= os.path.join(logfolder, title))
    writer.flush()
    current_best_val = 0
    for epoch in range(epochruns):  # Adjust number of epochs as needed
        model.train()
        running_loss = 0.0
        size = len(trainloader.dataset)
        for batch, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            # print(labels)
            #train the model
            outputs = model(images)
            #obtain and backpropagate loss
            loss = criterion(outputs, labels)
            loss.backward()
            
            #clip gradient to prevent exploding gradient problem
            if grad_clip:
                torch.nn.utils.clip_grad_norm(model.parameters(),5)
        
            #step optimizer
            optimizer.step()
            if use_scheduler:
                scheduler.step()
            running_loss += loss.item()
            loss, current = loss.item(), (batch + 1) * len(images)
            if current%batchsize == 0:
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
        #validate
        if epoch%10 == 0:
            model.eval()
            predicted_labels.clear()
            actual_labels.clear()
            train_loss = 0
            test_loss = 0
            with torch.no_grad():
                for images, labels in trainloader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images).to(device)
                    loss = criterion(outputs, labels)
                    train_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    predicted_labels.extend(predicted.cpu())
                    actual_labels.extend(labels.cpu())
                train_bal_acc = balanced_accuracy_score(torch.tensor(actual_labels),torch.tensor(predicted_labels))
                writer.add_scalar('BalancedAccuracy/Train', train_bal_acc, epoch)
                predicted_labels.clear()
                actual_labels.clear()    
                for images, labels in testloader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    predicted_labels.extend(predicted.cpu())
                    actual_labels.extend(labels.cpu())
            val_bal_acc = balanced_accuracy_score(torch.tensor(actual_labels),torch.tensor(predicted_labels))
            writer.add_scalar('Loss/Train', (train_loss/len(trainloader)), epoch) 
            writer.add_scalar('Loss/Test', (test_loss/len(testloader)), epoch)
            writer.add_scalar('BalancedAccuracy/Test', val_bal_acc, epoch)
            writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        
            # writer.flush()
        if np.isnan(running_loss):
            break
        if(val_bal_acc > current_best_val):
            class_labels = sorted(target_labels, key=lambda x: target_labels[x])
            matrix = confusion_matrix(actual_labels, predicted_labels)
            plt.clf()
            seaborn.heatmap(matrix, annot = True, fmt = 'd', xticklabels=class_labels, yticklabels= class_labels)
            plt.title(f'Epoch{epoch}\nTrain Balanced Acc:{(train_bal_acc):.5f}%\nTest Balanced Acc:{(val_bal_acc):.5f}%')
            plt.xlabel('Pred VesselType')
            plt.ylabel('True VesselType')
            plt.xticks()
            plt.savefig(os.path.join(confusionmatrixfolder,f'{title}.png'))

            torch.save({
                    'epoch': epochruns,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    }, os.path.join(modelfolder, f'{title}.pt'))
            current_best_val = val_bal_acc
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(trainloader)}")


                


    # Evaluate the model
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        predicted_labels.clear()
        actual_labels.clear()
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predicted_labels.extend(predicted.cpu())
            actual_labels.extend(labels.cpu())
        train_bal_acc=balanced_accuracy_score(torch.tensor(actual_labels),torch.tensor(predicted_labels))
        print(f"Balanced Accuracy on train set: {(train_bal_acc):.5f}%")

        predicted_labels.clear()
        actual_labels.clear()
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predicted_labels.extend(predicted.cpu())
            actual_labels.extend(labels.cpu())
    test_bal_acc=balanced_accuracy_score(actual_labels,predicted_labels)
    if(test_bal_acc > current_best_val):
        class_labels = sorted(target_labels, key=lambda x: target_labels[x])
        matrix = confusion_matrix(actual_labels, predicted_labels)
        plt.clf()
        seaborn.heatmap(matrix, annot = True, fmt = 'd', xticklabels=class_labels, yticklabels= class_labels)
        plt.title(f'Train Balanced Acc:{(train_bal_acc):.5f}%\nTest Balanced Acc:{(test_bal_acc):.5f}%')
        plt.xlabel('Pred VesselType')
        plt.ylabel('True VesselType')
        plt.xticks()
        plt.savefig(os.path.join(confusionmatrixfolder,f'{title}.png'))

        print(f"Accuracy on test set: {(100 * correct / total):.2f}%")
        print(f"Balanced Accuracy on test set: {(test_bal_acc):.5f}%")

        torch.save({
                'epoch': epochruns,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                }, os.path.join(modelfolder, f'{title}.pt'))

            
                