import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from whale_dataset import WhaleDataset
import torch.optim as optim
from torch.autograd import Variable
import pickle
import pandas as pd
from transforms import Rescale, ToTensor
from torchvision import transforms
import torch


# Hyper Parameters
num_epochs = 5
batch_size = 5
learning_rate = 0.001
is_gpu = True

pretrained_model = torchvision.models.vgg16(pretrained=True)
torchvision.models.res

class MyExtendedVGG(nn.Module):
    def __init__(self, pretrained_model):
        super(MyExtendedVGG, self).__init__()

        self.__NUM_CLASS = 447
        self.pretrained_model = pretrained_model
        self.last_layer = nn.Linear(1000, self.__NUM_CLASS)

    def forward(self, x):
        return self.last_layer(self.pretrained_model(x))


net = MyExtendedVGG(pretrained_model).double()

# for param in net.features.parameters():
#     param.requires_grad = False

train_dataset = WhaleDataset(csv_file='/data/train.csv',
                             root_dir='/data/imgs',
                             train=True,
                             transform=transforms.Compose([
                                       Rescale((224, 224)),
                                       ToTensor()
                                   ]))
test_dataset = WhaleDataset(csv_file='/data/sample_submission.csv',
                            root_dir='/data/imgs',
                            transform=transforms.Compose([
                                Rescale((224, 224)),
                                ToTensor()
                            ]))
train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size,
                         shuffle=False)
print("Done loading data")

# Uncomment to load pre-trained model
# net = pickle.load(open("net_baseline.p", 'rb'))

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
