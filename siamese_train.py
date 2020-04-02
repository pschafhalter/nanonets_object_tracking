import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import os
from siamese_dataloader import *
from siamese_net import *

import nonechucks as nc
from scipy.stats import multivariate_normal


class Config():
	training_dir = "/data/amandhar/crops/train/"
	testing_dir = "/data/amandhar/crops/test/"
	train_batch_size = 128
	train_number_epochs = 100

train_dataset = dset.ImageFolder(root=Config.training_dir)
test_dataset = dset.ImageFolder(root=Config.testing_dir)

transforms = torchvision.transforms.Compose([
	torchvision.transforms.Resize((128, 128)), # height, width
	torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
	torchvision.transforms.RandomHorizontalFlip(),
	torchvision.transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
	torchvision.transforms.ToTensor()
	])


def get_gaussian_mask():
	#128x64 is image size
	x, y = np.mgrid[0:1.0:128j, 0:1.0:128j]
	xy = np.column_stack([x.flat, y.flat])
	mu = np.array([0.5,0.5])
	sigma = np.array([0.35,0.25])
	covariance = np.diag(sigma**2) 
	z = multivariate_normal.pdf(xy, mean=mu, cov=covariance) 
	z = z.reshape(x.shape) 

	z = z / z.max()
	z  = z.astype(np.float32)

	mask = torch.from_numpy(z)
	return mask



siamese_train_set = SiameseTriplet(imageFolderDataset=train_dataset, transform=transforms, should_invert=False)
siamese_test_set = SiameseTriplet(imageFolderDataset=test_dataset, transform=transforms, should_invert=False)
net = SiameseNetwork().cuda()

criterion = TripletLoss(margin=1)
optimizer = optim.Adam(net.parameters(), lr = 0.0005) #changed from 0.0005

print(torch.cuda.is_available())

counter = []
loss_history = []
test_counter = []
test_loss_history = []
iteration_number = 0

train_dataloader = DataLoader(siamese_train_set, shuffle=True, num_workers=14, batch_size=Config.train_batch_size)
test_dataloader = DataLoader(siamese_test_set, shuffle=True, num_workers=14, batch_size=Config.train_batch_size)
num_test_batches = len(siamese_test_set) // test_dataloader.batch_size
#Multiply each image with mask to give attention to center of the image.
gaussian_mask = get_gaussian_mask().cuda()
best_test_loss, best_epoch = float('inf'), 0

for epoch in range(0, Config.train_number_epochs):
	# Train
	net.train()
	for i, train_batch in enumerate(train_dataloader, 0):
		anchor, positive, negative = train_batch
		anchor, positive, negative = anchor.cuda(), positive.cuda(), negative.cuda()
		anchor, positive, negative = anchor * gaussian_mask, positive * gaussian_mask, negative * gaussian_mask
		optimizer.zero_grad()
		anchor_out, positive_out, negative_out = net(anchor, positive, negative)
		triplet_loss = criterion(anchor_out, positive_out, negative_out)
		triplet_loss.backward()
		optimizer.step()
		# Record training loss every 10 iterations
		if i % 10 == 0:
			print("Epoch number: {}\n Current training loss: {}\n".format(epoch, triplet_loss.item()))
			iteration_number += 10
			counter.append(iteration_number)
			loss_history.append(triplet_loss.item())
	# Test/Validate
	test_losses = []
	net.eval()
	for i, test_batch in enumerate(test_dataloader, 0):
		anchor, positive, negative = test_batch
		anchor, positive, negative = anchor.cuda(), positive.cuda(), negative.cuda()
		anchor, positive, negative = anchor * gaussian_mask, positive * gaussian_mask, negative * gaussian_mask
		anchor_out, positive_out, negative_out = net(anchor, positive, negative)
		test_triplet_loss = criterion(anchor_out, positive_out, negative_out)
		test_losses.append(test_triplet_loss.item())
	test_counter.append(iteration_number)
	mean_test_loss = np.mean(test_losses)
	print("Mean test loss: {}\n".format(mean_test_loss))
	test_loss_history.append(mean_test_loss)
	# Save model every epoch
	if not os.path.exists('ckpts/'):
		os.mkdir('ckpts')
	torch.save(net, 'ckpts/model' + str(epoch) + '.pt')
	torch.save(net.state_dict(), 'ckpts/model' + str(epoch) + '_state-dict')
	if mean_test_loss < best_test_loss:
		best_test_loss = mean_test_loss
		best_epoch = epoch
		torch.save(net, 'ckpts/best_model.pt')
		torch.save(net.state_dict(), 'ckpts/best_model_state-dict')
		print("BEST EPOCH updated to {}".format(best_epoch))
	show_plot(counter, loss_history, path='ckpts/train_loss.png')
	show_plot(test_counter, test_loss_history, path='ckpts/test_loss.png')
