import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

import noise

import sys
sys.path.append('./model')
import resnet

torch.manual_seed(0)

display = False

model_dir = '../models'
model_file = 'resnet18/final_10norm_1500.pth'
model = resnet.ResNet18()

noise_type = 'uniform'
unif_min = 0
unif_max = 1
gaussian_mean = 0
gaussian_std = 0.05

noise_options = ['uniform', 'gaussian']

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('CUDA?', device)

# Data
print('==> Preparing test data..')
print('Noise:', noise_type)

if noise_type == 'uniform':
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        noise.AddUniformNoise(unif_min, unif_max)
    ])
elif noise_type == 'gaussian':
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        noise.AddGaussianNoise(gaussian_mean, gaussian_std)
    ])
else:
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if display:
    def imshow(img):
        img[0] = img[0] * 0.247 + 0.4914     # unnormalize
        img[1] = img[1] * 0.243 + 0.4822
        img[2] = img[2] * 0.261 + 0.4465
        npimg = img.numpy()   # convert from tensor
        npimg = np.clip(npimg, 0, 1)
        plt.imshow(np.transpose(npimg, (1, 2, 0))) 

    print('...displaying...')

    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(testset), size=(1,)).item()
        img, label = testset[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(classes[label])
        plt.axis("off")
        imshow(torchvision.utils.make_grid(img))

    if noise_type in noise_options:
        plt.savefig('cifar10_' + noise_type + '.png')
    else:
        plt.savefig('cifar10' + '.png')


# Model
print('==> Building model..')

if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True
model = model.cuda()

model.load_state_dict(torch.load(model_dir + '/' + model_file))

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in testloader:
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        
        images = Variable(images)
        labels = Variable(labels)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))

