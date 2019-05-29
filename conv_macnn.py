import time
import torch.optim
import torch.nn as nn
from torchvision import models
from torch.autograd import Variable


# 超参数
learning_rate = 1e-3
input_channels = 1
output_features = 6
epoch = 1
save_model_name = 'conv.pth'


# 卷积网络 
conv = models.resnet18().cuda()
conv.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).cuda()
conv.fc = nn.Linear(in_features=512, out_features=output_features, bias=True).cuda()

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(conv.parameters(), lr=learning_rate)

for epoch_number in range(epoch):
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
    print(epoch_number, count, running_loss, int(acc)/count)

torch.save(conv, save_model_name)

count, acc = 0, 0.
for data in test_loader:
    img, label = data
    img = Variable(img).cuda()
    label = Variable(label).cuda()
    output = conv(img)
    acc += (torch.max(output, dim=1)[1] == label).sum()
    count += img.size(0)
print(count, running_loss, int(acc)/count)
