import cv2
import time
import h5py
import torch.optim
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


class Part(nn.Module):

    def __init__(self):
        super(Part, self).__init__()
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)

    def forward(self, x):
        conv_matrix = torch.clone(x)
        conv_matrix = conv_matrix.reshape(conv_matrix.size(0), 256, 1, 784)
        conv_matrix = conv_matrix.transpose(1, 3)
        x = F.avg_pool2d(x, kernel_size=28, stride=28)
        x = x.view(x.size(0), -1)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        x = torch.sigmoid(x).unsqueeze(1).unsqueeze(1)
        x = F.interpolate(x, (1, 784), mode='bilinear', align_corners=True)
        x = x.squeeze(1).squeeze(1).unsqueeze(2).unsqueeze(3)
        x = x * conv_matrix
        x = F.avg_pool2d(x, kernel_size=(1, 512), stride=512)
        x = x * 0.1
        x = F.softmax(x, dim=1)
        x = torch.exp(x)
        x = x + 1
        x = torch.log(x)
        x = x * 4
        x = x.squeeze(2).squeeze(2)
        return x.reshape(x.size(0), 28, 28)


class Clf(nn.Module):

    def __init__(self):
        super(Clf, self).__init__()
        self.res1 = models.resnet18()
        self.res1.conv1 = nn.Conv2d(256, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).cuda()
        self.res1.fc = nn.Linear(in_features=512, out_features=6, bias=True).cuda()
        self.res2 = models.resnet18()
        self.res2.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).cuda()
        self.res2.fc = nn.Linear(in_features=512, out_features=6, bias=True).cuda()

    def forward(self, channels, attention):
        xc = self.res1(channels)
        xa = self.res2(attention)
        return F.softmax(xc + xa, dim=1)


def get_channels(c, data):
    return c.layer3(c.layer2(c.layer1(c.maxpool(c.relu(c.bn1(c.conv1(data)))))))


class BU3DDataset(Dataset):
    def __init__(self, x, y):
        super(BU3DDataset, self).__init__()
        self.x = x
        self.y = y

    def __getitem__(self, index):
        img = torch.tensor(cv2.resize(x[index], (448, 448))).float().unsqueeze(0)
        label = y[index]
        return img, label

    def __len__(self):
        return len(self.x)


f = h5py.File('bu3d_features.h5')
x, y = [], []
for index, name in enumerate(f):
    for file in f[name]:
        x.append(f[name][file].value)
        y.append(index)
f.close()

batch_size = 10
learning_rate = 1e-3

x, tx, y, ty = train_test_split(x, y, test_size=106/606, random_state=0)

train_set = BU3DDataset(x, y)
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_set = BU3DDataset(tx, ty)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

clf = Clf().cuda()

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.RMSprop(clf.parameters(), lr = learning_rate)

conv = torch.load('conv.pth')
part = torch.load('part.pth')

epoch = 0
while True:
    for data in train_loader:
        img, label = data
        img = Variable(img).cuda()
        label = Variable(label).cuda()
        channels = get_channels(conv, img)
        attention = part(channels).unsqueeze(1)
        output = clf(channels, attention)
        optimizer.zero_grad()
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()
    count, acc = 0, 0.
    for data in test_loader:
        img, label = data
        img = Variable(img).cuda()
        label = Variable(label).cuda()
        channels = get_channels(conv, img)
        attention = part(channels).unsqueeze(1)
        output = clf(channels, attention)
        acc += (torch.max(output, dim=1)[1]==label).sum()
        count += img.size(0)
    print(epoch, (int(acc)/count)*100,'%', time.asctime())
    if (int(acc)/count)*100 > 95.:
        torch.save(clf, 'clfd.pth')
        break
    epoch += 1

chaos_matrix = torch.zeros((6, 6))
for data in test_loader:
    img, label = data
    img = Variable(img).cuda()
    label = Variable(label).cuda()
    channels = get_channels(conv, img)
    attention = part(channels).unsqueeze(1)
    output = clf(channels, attention)
    output = torch.max(output, dim=1)[1]
    for (ix, iy) in zip(output, label):
        chaos_matrix[ix, iy] += 1

chaos_matrix