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


class Loss(nn.Module):

    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, tensor):
        loss_sum = torch.zeros(1).cuda()
        indexes = Loss.get_max_index(tensor)
        for i in range(len(indexes)):
            max_x, max_y = indexes[i]
            for j in range(tensor.size(1)):
                for k in range(tensor.size(2)):
                    loss_sum += ((max_x - j) * (max_x - j) + (max_y - k) * (max_y - k)) * tensor[i, j, k]
        return loss_sum

    @staticmethod
    def get_max_index(tensor):
        shape = tensor.shape
        indexes = []
        for i in range(shape[0]):
            mx = tensor[i, 0, 0]
            x, y = 0, 0
            for j in range(shape[1]):
                for k in range(shape[2]):
                    if tensor[i, j, k] > mx:
                        mx = tensor[i, j, k]
                        x, y = j, k
            indexes.append([x, y])
        return indexes

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

batch_size = 4
learning_rate = 1e-3

x, tx, y, ty = train_test_split(x, y, test_size=1/6, random_state=0)

train_set = BU3DDataset(x, y)
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_set = BU3DDataset(tx, ty)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

conv = torch.load('conv.pth')

part = Part().cuda()

loss_fn = Loss()

optimizer = torch.optim.Adam(part.parameters(), lr = learning_rate)

for epoch in range(1):
    running_loss, count, acc = 0., 0, 0.
    print(time.asctime())
    for data in train_loader:
        img, _ = data
        img = Variable(img).cuda()
        channels = get_channels(conv, img)
        output = part(channels)
        optimizer.zero_grad()
        loss = loss_fn(output)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        count += img.size(0)
    print(epoch, count, running_loss, Loss.get_max_index(output))


torch.save(part, 'part.pth')

count=0
for data in test_loader:
    img, _ = data
    img = Variable(img).cuda()
    channels = get_channels(conv, img)
    output = part(channels)
    count += img.size(0)
    print(count, Loss.get_max_index(output))

