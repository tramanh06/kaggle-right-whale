from __future__ import print_function, division
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
from transforms import Rescale, ToTensor
from whale_dataset import WhaleDataset

# Hyper Parameters
num_epochs = 2
batch_size = 5
learning_rate = 0.001
is_gpu = True


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.__NUM_CLASS = 447
        # self.__NUM_CLASS = 10
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(32 * 64 * 96, self.__NUM_CLASS)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


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
# optimizer = optim.Adam(net.parameters(), lr=learning_rate)
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9) # This one works better than Adam
print("Done loading optimizer")

for epoch in range(num_epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
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
        if i % 1 == 0:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.9f' %
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
    # print("Images glimpse: ")
    # print(images[0])
    if is_gpu:
        images = Variable(images.cuda())
    else:
        images = Variable(images)
    outputs = net(images)
    # print("Output:")
    # print(outputs)
    _, predicted = torch.max(outputs.data, 1)
    print("Predicted")
    print(predicted)

    if is_gpu:
        whale_id = test_dataset.inverse_transform(encoder, predicted.cpu().numpy())
    else:
        whale_id = test_dataset.inverse_transform(encoder, predicted.numpy())
    print("Images")
    print(image_names)
    print("Corresponding whale ID")
    print(whale_id)

    dictionary = list(zip(image_names, whale_id))
    test_data_list.extend(dictionary)

print(test_data_list)
submission = pd.read_csv('/data/sample_submission.csv')
# whale_00195 - set all value at this column to 0
second_column = 'whale_00195'
submission.loc[:, second_column] = 0
for (name, whale_id) in test_data_list:
    submission.loc[submission.Image == name, whale_id] = 1
submission.to_csv("/output/submission.csv", index=False)
