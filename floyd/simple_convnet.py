from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from sklearn import preprocessing
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle

# Hyper Parameters
num_epochs = 5
batch_size = 20
learning_rate = 0.001
is_gpu = True


class WhaleDataset(Dataset):
    """Whale dataset."""

    encoder_filepath = "label_encoder.p"

    def __init__(self, csv_file, root_dir, train=False, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        label_data = pd.read_csv(csv_file)
        label_encoder = preprocessing.LabelEncoder()
        if train:
            label_data['label'] = label_encoder.fit_transform(label_data['whaleID'])
            pickle.dump(label_encoder, open(self.encoder_filepath, "wb"))
            print("Finish writing encoder to file")
        else:
            label_data['label'] = -1

        self.img_lookup = label_data
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_lookup)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.img_lookup.ix[idx, 0])
        image = io.imread(img_name)

        whale_id = self.img_lookup.ix[idx, 2]
        sample = {'image_name': self.img_lookup.ix[idx, 0],
                  'image': image,
                  'whale_id': whale_id}

        if self.transform:
            sample = self.transform(sample)

        return sample

    @staticmethod
    def inverse_transform(encoder, class_labels):
        return encoder.inverse_transform(class_labels)

    def get_encoder(self):
        return pickle.load(open(self.encoder_filepath, "rb"))


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
        name, image, whale_id = sample['image_name'],sample['image'], sample['whale_id']

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

        return {'image_name': name, 'image': img, 'whale_id': whale_id}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image_name,image, whale_id = sample['image_name'],sample['image'], sample['whale_id']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image_name': image_name,
                'image': torch.from_numpy(image),
                'whale_id': whale_id}


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
        self.fc1 = nn.Linear(16 * 61 * 93, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.__NUM_CLASS)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 61 * 93)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


train_dataset = WhaleDataset(csv_file='/data/train.csv',
                             root_dir='/data/imgs',
                             train=True,
                             transform=transforms.Compose([
                                       Rescale((256, 384)),
                                       ToTensor()
                                   ]))
test_dataset = WhaleDataset(csv_file='/data/sample_submission.csv',
                            root_dir='/data/imgs',
                            transform=transforms.Compose([
                                Rescale((256, 384)),
                                ToTensor()
                            ]))
train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size,
                         shuffle=False)
print("Done loading data")

# Uncomment to load pre-trained model
# net = pickle.load(open("net_baseline.p", 'rb'))
net = Net().double()
if is_gpu:
    net.cuda()
print("Done loading net")

criterion = nn.CrossEntropyLoss()
print("Done loading loss")
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
print("Done loading optimizer")

for epoch in range(num_epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        print("Batch No. %d" % i)
        # get the inputs
        inputs, labels = data['image'], data['whale_id']

        # wrap them in Variable
        if is_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 10 == 0:  # print every 2000 mini-batches
            print('[%d, %9d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

pickle.dump(net, open("/output/net_baseline.p", "wb"))

# Test the model
net.eval()

encoder = test_dataset.get_encoder()
test_data_list = []

for data in test_loader:
    images = data['image']
    image_names = data['image_name']
    print("Images glimpse: ")
    print(images[0])
    if is_gpu:
        images = Variable(images.cuda())
    else:
        images = Variable(images)
    outputs = net(images)
    print("Output:")
    print(outputs)
    _, predicted = torch.max(outputs.data, 1)
    print("Predicted")
    print(predicted)

    if is_gpu:
        whale_id = test_dataset.inverse_transform(encoder, predicted.cpu().numpy())
    else:
        whale_id = test_dataset.inverse_transform(encoder, predicted.numpy())
    print("Corresponding whale ID")
    print(whale_id)

    dictionary = list(zip(image_names, whale_id))
    test_data_list.extend(dictionary)

# Write to submission file
predicted_compiled = pd.DataFrame(columns=['Image', 'whale_id'], data=test_data_list)
one_hot = pd.get_dummies(predicted_compiled['whale_id'])
# Drop column whale_id as it is now encoded
predicted_compiled = predicted_compiled.drop('whale_id', axis=1)
# Join the encoded df
for_submission = predicted_compiled.join(one_hot)
for_submission.to_csv("/output/submission.csv", index=False)