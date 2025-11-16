import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

class EmberNet(nn.Module):
	def __init__(self):
		super(EmberNet, self).__init__()
		self.relu = nn.ReLU()
		# layer1 convolution: 18 24*24 channels
		self.conv1 = nn.Conv2d(in_channels=3, out_channels=18, kernel_size=9, stride=1, bias=False)
		# layer2 subsampling: 18 12*12 channels
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
		# layer3 convolution: 32 10*10 channels
		self.conv3 = nn.Conv2d(in_channels=18, out_channels=32, kernel_size=3, stride=1, bias=False)
		# layer4 subsampling: 32 5*5 channels
		self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
		# layer5: 800 inputs, 120 outputs=
		#self.conv5 = nn.conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, bias=False)
		self.dropout = nn.Dropout(0.2)
		self.layer5 = nn.Linear(800, 120)
		# layer6: 120 inputs, 84 outputs
		self.layer6 = nn.Linear(120, 84)
		# layer7: 84 inputs, 10 outputs
		self.layer7 = nn.Linear(84, 10)

	def forward(self, x):
		x = self.relu(self.conv1(x))
		x = self.pool2(x)
		x = self.relu(self.conv3(x))
		x = self.pool4(x)

		x = x.view(x.size(0), -1)
		x = self.relu(self.layer5(x))
		x = self.dropout(x)
		x = self.relu(self.layer6(x))
		x = self.dropout(x)
		x = self.layer7(x)

		return x

def load_data(data_root, batch_size = 32, trainset_size = 0):
	
	# load training data from CIFAR10
	train_set = datasets.CIFAR10(data_root, train=True, download=True, transform=transforms.ToTensor())
	
	if trainset_size != 0:
		train_set, other_set = torch.utils.data.random_split(train_set, [trainset_size, len(train_set) - trainset_size])
		
	# load testing data
	test_set = datasets.CIFAR10(data_root, train=False, download=True, transform=transforms.ToTensor())

	# dataloaders
	# shuffle data for SGD - ensure model is learning and not memorizing dataset
	train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
	test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True)

	print("Number of images in train dataset:", len(train_set))
	print("Number of images in test dataset:", len(test_set))

	return train_loader, test_loader

def train(data_loader, model, criteria, optimizer, epoch, device="cpu"):
	all_losses = []
	correct = 0
	print_freq = int(len(data_loader)/20) + 1
	for batch_id, (inputs, targets) in enumerate(data_loader):
		inputs, targets = inputs.to(device), targets.to(device)
		
		optimizer.zero_grad()
		outputs = model(inputs)
		loss = criteria(outputs, targets)
		loss.backward()
		optimizer.step()
		
		all_losses.append(loss)
		pred = outputs.data.max(1, keepdim=True)[1]
		correct += pred.eq(targets.data.view_as(pred)).cpu().sum()
		if batch_id % print_freq == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_id * len(inputs), len(data_loader.dataset),
				100. * batch_id / len(data_loader), loss.data.item()))
			
	train_acc = 100. * correct / len(data_loader.dataset)
	train_loss = float(sum(all_losses))/len(data_loader.dataset)
	print("Train set: Average loss: {:.4f}, Accuracy {}/{} ({:.2f}%)".format(
		train_loss, correct, len(data_loader.dataset), train_acc))
	return train_loss, train_acc

def inference(data_loader, model, criteria, device='cpu'):
	model.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for batch_id, (inputs, targets) in enumerate(data_loader):
			inputs, targets = inputs.to(device), targets.to(device)
			outputs = model(inputs)
			loss = criteria(outputs, targets)
			test_loss += loss

			pred = outputs.data.max(1, keepdim=True)[1]  # get the index of the class with the maximum log-probability
			correct += pred.eq(targets.data.view_as(pred)).cpu().sum()
	test_acc = 100. * correct / len(data_loader.dataset)
	test_loss /= len(data_loader.dataset)
	print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
		test_loss, correct, len(data_loader.dataset),test_acc))
	return test_loss,test_acc

def execute(net, epochs, trainset_size):
	lr = 0.01
	batch_size = 32
	criteria = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(net.parameters(), lr=lr)
	train_loader, test_loader = load_data(data_root = "./", batch_size=batch_size, trainset_size=trainset_size)
	
	print("-"*20, "network training", "-"*20)
	
	train_loss, train_acc = [], []
	test_loss, test_acc = [], []
	
	for epoch in range(0, epochs):
		# train model
		tr_loss, tr_acc = train(train_loader, net, criteria, optimizer, epoch)
		te_loss, te_acc = inference(test_loader, net, criteria)

		train_loss.append(tr_loss)
		train_acc.append(tr_acc)
		test_loss.append(te_loss)
		test_acc.append(te_acc)
		
	print("-"*20, "loss and accuracy", "-"*20)
	
	plt.figure(figsize=(3,5))
	plt.plot(range(1, epochs + 1), train_acc)
	plt.plot(range(1, epochs + 1), test_acc)
	plt.title("Train Accuracy using {} training data in {} epochs".format(trainset_size, epochs))
	
	plt.figure(figsize=(3,5))
	plt.plot(range(1, epochs + 1), train_loss)
	plt.plot(range(1, epochs + 1), test_loss)
	plt.title("Train Loss using {} training data in {} epochs".format(trainset_size, epochs))
	plt.show()
	
	return train_loss, train_acc, test_loss, test_acc

net = EmberNet()
# train and test EmberNet model for 50 epochs with 50000 images
execute(net, 50, 50000)

# save as ONNX for OpenCV processing
torch.save(net.state_dict(), "EmberNet.pth")
net.eval()
net.load_state_dict(torch.load("EmberNet.pth", map_location="cpu"))
net.eval()

CIFAR10_size = torch.rand(1, 3, 32, 32)

torch.onnx.export(
    net,
    CIFAR10_size,
    "EmberNet.onnx",
    input_names=["input"],
    output_names=["output"]
)