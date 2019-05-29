import time
import torch.optim
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from data_macnn import test_loader
from data_macnn import train_loader
from torch.autograd import Variable


# 超参数
target_accuracy = 95.
learning_rate = 1e-3
input_channels = 1
output_features = 6
save_clf_name = 'clf.pth'
part_model_name = 'part.pth'
conv_model_name = 'conv.pth'


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
        self.res1.fc = nn.Linear(in_features=512, out_features=output_features, bias=True).cuda()
        self.res2 = models.resnet18()
        self.res2.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).cuda()
        self.res2.fc = nn.Linear(in_features=512, out_features=output_features, bias=True).cuda()

    def forward(self, channels, attention):
        xc = self.res1(channels)
        xa = self.res2(attention)
        return F.softmax(xc + xa, dim=1)


def get_channels(c, data):
    return c.layer3(c.layer2(c.layer1(c.maxpool(c.relu(c.bn1(c.conv1(data)))))))


clf = Clf().cuda()

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.RMSprop(clf.parameters(), lr = learning_rate)

conv = torch.load(conv_model_name)
part = torch.load(part_model_name)

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
    if (int(acc)/count)*100 > target_accuracy:
        torch.save(clf, save_clf_name)
        break
    epoch += 1

chaos_matrix = torch.zeros((output_features, output_features))
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

print(chaos_matrix)
