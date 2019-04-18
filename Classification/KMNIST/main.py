from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from kmnist import kmnist



log_interval=200
seed=1
momentum=0
lr=0.01
epochs = 10 
batch_size=64 

use_cuda = torch.cuda.is_available()

torch.manual_seed(seed)
if use_cuda:
    torch.cuda.manual_seed(seed)



train_loader = torch.utils.data.DataLoader(kmnist('./data', train=True, download=True, transform=transforms.ToTensor()), batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(kmnist('./data', train=False, download=True, transform=transforms.ToTensor()), batch_size=batch_size, shuffle=True)




class Net(nn.Module):
	def __init__(self, hidden_size=400):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(28 * 28, hidden_size)
		self.fc3 = nn.Linear(hidden_size, hidden_size)
		self.BN = nn.BatchNorm1d(hidden_size)
		self.fc4 = nn.Linear(hidden_size, 10)

	def forward(self, input):

		input=input.view(-1,784)

		x = F.relu(self.fc1(input))
		x = F.relu(self.fc3(x))
		x = self.BN(x)
		x = F.relu(self.fc4(x))
		return F.log_softmax(x)

model = Net()


if use_cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * data.size(0), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test():
	model.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for data, target in test_loader:
			if use_cuda:
				data, target = data.cuda(), target.cuda()
			data, target = Variable(data), Variable(target)
			output = model(data)
			test_loss += F.nll_loss(output, target, size_average=False).item()
			pred = output.data.max(1, keepdim=True)[1]
			correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

		test_loss /= len(test_loader.dataset)
		print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
			test_loss, correct, len(test_loader.dataset),
			100. * correct / len(test_loader.dataset)))


for epoch in range(1, epochs + 1):
    train(epoch)
    test()
