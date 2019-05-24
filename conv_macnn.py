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

conv = models.resnet18().cuda()
conv.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).cuda()
conv.fc = nn.Linear(in_features=512, out_features=6, bias=True).cuda()

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(conv.parameters(), lr = learning_rate)

for epoch in range(8):
    running_loss, count, acc = 0., 0, 0.
    print(time.asctime())
    for data in train_loader:
        img, label = data
        img = Variable(img).cuda()
        label = Variable(label).cuda()
        output = conv(img)
        optimizer.zero_grad()
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        acc += (torch.max(output, dim=1)[1]==label).sum()
        count += img.size(0)
    print(epoch, count, running_loss, int(acc)/count)

torch.save(conv, 'conv.pth')

count, acc = 0, 0.
for data in test_loader:
    img, label = data
    img = Variable(img).cuda()
    label = Variable(label).cuda()
    output = conv(img)
    acc += (torch.max(output, dim=1)[1]==label).sum()
    count += img.size(0)
print(count, running_loss, int(acc)/count)