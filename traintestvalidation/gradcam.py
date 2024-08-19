import torch
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch.utils.data import DataLoader
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet101
import rasterio as rio
from rasterio.plot import show
import pandas as pd
from ship_classification import groupLabels, GeotiffDataset, getMeanandStd
from sklearn.model_selection import train_test_split
from torchvision.transforms import v2
import matplotlib
import matplotlib.pyplot as plt
import os
from importlib import reload


checkpointpath = 'models/cresnet101p_SGD_epoch300_lr0.0001_oversampler_cosineannealingwarmrestarttmax50labelsmooth0.1_weightdecay0.0001_11kdatasetnormed_augmentedaggressive_.pt'
imagefolder = 'dso_internship'
train = pd.read_csv('11kdatasetdualchannel/train.csv', index_col= [0]).reset_index()
test = pd.read_csv('11kdatasetdualchannel/test.csv', index_col=[0]).reset_index()
outputfolder = 'traintestvalidation/11kdatasetdualchannel/labelsmoothed0.1_all'
if not os.path.exists(outputfolder):
    os.mkdir(outputfolder)
mean, std = getMeanandStd(train, imagefolder)
batch_size = 100
cropsize= 128

checkpoint = torch.load(checkpointpath)
model = resnet101(weights = 'DEFAULT')
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 5) #change the output depending on number of model classification labels
model.load_state_dict(checkpoint['model_state_dict'])


transform_list = [
        v2.Normalize(mean = mean, std=std),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.CenterCrop(cropsize), #CenterCrop image to cropsize
        # v2.Lambda(lambda x: x.repeat(3,1,1))#for single channel uncomment
    ]
transform = v2.Compose(transform_list)

testdataset = GeotiffDataset(test, imagefolder= imagefolder, transform= transform, augment = False)
testloader = DataLoader(testdataset, batch_size=batch_size, shuffle=False)

#set device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# targetlayer = [model.layer4[-1]] #take the gradients of the last activation layer 
targetlayer = [model.layer1[-1], model.layer2[-1], model.layer3[-1], model.layer4[-1]] #take the average of all layers
targets = None

incorrect_sample = []
incorrect_label = []
incorrect_pred = []
correct_sample = []

model.to(device)
model.eval()
for batch, (images, labels) in enumerate(testloader):
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    for sampleno in range(images.shape[0]):
        imagename = f"batch{batch}_sample{sampleno}_actual{labels[sampleno]}_pred{predicted[sampleno]}"
        img = images[sampleno][0].cpu().numpy()
        inputtensor = images[sampleno].unsqueeze(0)
        with GradCAM(model = model, target_layers = targetlayer) as cam:
            grayscale_cam = cam(input_tensor=inputtensor, targets=targets)
            grayscale_cam = grayscale_cam[0,:]
            fig = plt.figure()
            fig.suptitle(f'Actual Label: {labels[sampleno]}\nPredicted Label: {predicted[sampleno]}')
            ax1 = plt.subplot(121)
            plt.imshow(img, cmap='gray')
            ax2 = plt.subplot(122, sharex = ax1, sharey = ax1)
            plt.imshow(grayscale_cam)
            # plt.show()
            fig.savefig(os.path.join(outputfolder, imagename))
            plt.clf()
    reload(matplotlib)
    matplotlib.use('Agg')
    