import time
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from data_macnn import test_loader
from data_macnn import train_loader
from torch.autograd import Variable


# 超参数
epoch = 1
learning_rate = 1e-3
save_part_name = 'part.pth'
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
        '''
        # Div
        mgr = 0.
        for data in x:
            mgr += data.mean()
        mgr /= len(x)
        for i in range(x[0].size(0)):
            for j in range(x[0].size(1)):
                for k in range(x[0].size(2)):
                    tensors = []
                    for tensor in x:
                        tensors.append(float(tensor[i, j, k]))
                    for r in range(len(tensors)):
                        loss_sum += tensors[r] * (max(tensors[:r] + tensors[r + 1:]) - mgr)
        '''
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


conv = torch.load(conv_model_name)

part = Part().cuda()

loss_fn = Loss()

optimizer = torch.optim.Adam(part.parameters(), lr = learning_rate)

for epoch_number in range(epoch):
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
    print(epoch_number, count, running_loss, Loss.get_max_index(output))


torch.save(part, save_part_name)

count=0
for data in test_loader:
    img, _ = data
    img = Variable(img).cuda()
    channels = get_channels(conv, img)
    output = part(channels)
    count += img.size(0)
    print(count, Loss.get_max_index(output))

