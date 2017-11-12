from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from config import FILEPATH_CONFIG
from sklearn import preprocessing


class WhaleDataset(Dataset):
    """Whale dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        label_data = pd.read_csv(csv_file)
        label_encoder = preprocessing.LabelEncoder()
        label_data['label'] = label_encoder.fit_transform(label_data['whaleID'])
        # label_data.to_csv("labels.csv")

        self.img_lookup = label_data
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_lookup)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.img_lookup.ix[idx, 0])
        image = io.imread(img_name)
        whale_id = self.img_lookup.ix[idx, 2]
        sample = {'image': image, 'whale_id': whale_id}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_number_of_class(self):
        return self.img_lookup['label'].max(axis=0)

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, whale_id = sample['image'], sample['whale_id']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
#         landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'whale_id': whale_id}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, whale_id = sample['image'], sample['whale_id']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        return {'image': image, 'whale_id': whale_id}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, whale_id = sample['image'], sample['whale_id']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'whale_id': whale_id}

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.__NUM_CLASS = 447
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6,
                               kernel_size=3, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Conv1: 256x384x3 => 254x382x6
        # Pool1: 254x382x6 => 127x191x6
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16,
                               kernel_size=5)
        # Conv2: 127x191x6 => 123x187x16
        # Pool2: 123x187x16 => 61x93x16
        self.fc1 = nn.Linear(16*61*93, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.__NUM_CLASS)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # print("After conv2:")
        # print(x.size())
        x = x.view(-1, 16*61*93)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # print("After FC3:")
        # print(x.size())
        return x


# whale_dataset = WhaleDataset(csv_file=FILEPATH_CONFIG['data']+'train.csv',
#                             root_dir=FILEPATH_CONFIG['data']+'imgs')

transformed_dataset = WhaleDataset(csv_file=FILEPATH_CONFIG['data']+'train.csv',
                                           root_dir=FILEPATH_CONFIG['data']+'imgs',
                                           transform=transforms.Compose([
                                               Rescale((256,384)),
                                               ToTensor()
                                           ]))
dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=False, num_workers=4)

net = Net().double()

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        # get the inputs
        inputs, labels = data['image'], data['whale_id']

        # print("Input image size:")
        # print(inputs.size())

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        # print(outputs)
        loss = criterion(outputs, labels)
        print(loss)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
              (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')