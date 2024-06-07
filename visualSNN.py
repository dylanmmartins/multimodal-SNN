import snntorch as snn
import torch
from torchvision import datasets, transforms
from snntorch import utils
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from snntorch import spikegen
import numpy as np

from noise_helpers import AddGaussianNoise

dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
batch_size = 128*4

# Define Network
class VisNet(nn.Module):
	def __init__(self, num_inputs, num_hidden=1000, num_last_hidden=20, num_output=10,
				 num_steps=81, beta=0.95):
		super().__init__()

		self.num_steps = num_steps

		# Initialize layers
		self.fc1 = nn.Linear(num_inputs, num_hidden)
		self.lif1 = snn.Leaky(beta=beta)
		self.fc2 = nn.Linear(num_hidden, num_hidden)
		self.lif2 = snn.Leaky(beta=beta)
		self.fc3 = nn.Linear(num_hidden, num_last_hidden)
		self.lif3 = snn.Leaky(beta=beta)
		self.fc4 = nn.Linear(num_last_hidden, num_output)
		self.lif4 = snn.Leaky(beta=beta)


	def forward(self, x):

		# Initialize hidden states at t=0
		mem1 = self.lif1.init_leaky()
		mem2 = self.lif2.init_leaky()
		mem3 = self.lif3.init_leaky()
		mem4 = self.lif4.init_leaky()

		# Record the output layer
		output_spike_record = []
		output_memV_record = []


		for step in range(self.num_steps):

			cur1 = self.fc1(x)
			spk1, mem1 = self.lif1(cur1, mem1)

			cur2 = self.fc2(spk1)
			spk2, mem2 = self.lif2(cur2, mem2)

			cur3 = self.fc3(spk2)
			spk3, mem3 = self.lif3(cur3, mem3)

			cur4 = self.fc4(spk3)
			spk4, mem4 = self.lif4(cur4, mem4)

			output_spike_record.append(spk4)
			output_memV_record.append(mem4)

		self.mem1 = mem1
		self.mem2 = mem2
		self.mem3 = mem3

		return  torch.stack(output_spike_record, dim=0), torch.stack(output_memV_record, dim=0)

	def fwd_frozen(self, x):

		mem1 = self.lif1.init_leaky()
		mem2 = self.lif2.init_leaky()
		mem3 = self.lif3.init_leaky()

		last_hidden_spike_record = []
		last_hidden_output_memV_record = []

		for step in range(self.num_steps):

			cur1 = self.fc1(x)
			spk1, mem1 = self.lif1(cur1, mem1)

			cur2 = self.fc2(spk1)
			spk2, mem2 = self.lif2(cur2, mem2)

			cur3 = self.fc3(spk2)
			spk3, mem3 = self.lif3(cur3, mem3)

			last_hidden_spike_record.append(spk3)
			last_hidden_output_memV_record.append(mem3)

		return torch.stack(last_hidden_spike_record, dim=0), torch.stack(last_hidden_output_memV_record, dim=0)


def train_test_visual(savepath=None):

	# Dataloader arguments

	data_path='/tmp/data/mnist'

	# Define a transform
	transform = transforms.Compose([
				transforms.Resize((28, 28)),
				transforms.Grayscale(),
				transforms.ToTensor(),
				# AddGaussianNoise(0., 1.), # Add noise to the data
				transforms.Normalize((0,), (1,))])

	mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
	mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

	# Create DataLoaders
	train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
	test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)

	# Network Architecture
	num_inputs = 28*28

	num_epochs = 2
	loss_hist = []
	test_loss_hist = []
	counter = 0

	# Load the network onto CUDA if available
	net = VisNet(num_inputs=28*28).to(device)

	loss = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))

	# Outer training loop
	for epoch in range(num_epochs):
		iter_counter = 0
		train_batch = iter(train_loader)

		# Minibatch training loop
		for data, targets in train_batch:
			data = data.to(device)
			targets = targets.to(device)

			# forward pass
			net.train()
			spk_rec, mem_rec = net(data.view(batch_size, -1))

			# initialize the loss & sum over time
			loss_val = torch.zeros((1), dtype=dtype, device=device)
			for step in range(net.num_steps):
				loss_val += loss(mem_rec[step], targets)

			# Gradient calculation + weight update
			optimizer.zero_grad()
			loss_val.backward()
			optimizer.step()

			# Store loss history for future plotting
			loss_hist.append(loss_val.item())

			# Test set
			with torch.no_grad():
				net.eval()
				test_data, test_targets = next(iter(test_loader))
				test_data = test_data.to(device)
				test_targets = test_targets.to(device)

				# Test set forward pass
				#test_spk, test_mem = net(test_data.view(batch_size, -1))
				test_spk, test_mem = net(test_data)
				# Test set loss
				test_loss = torch.zeros((1), dtype=dtype, device=device)
				for step in range(net.num_steps):
					test_loss += loss(test_mem[step], test_targets)
				test_loss_hist.append(test_loss.item())

				# Print train/test loss/accuracy
				if counter % 50 == 0:

					print(f"Epoch {epoch}, Iteration {iter_counter}, Train loss = {loss_hist[counter]:.2f} Test loss = {test_loss_hist[counter]:.2f} \n")

					#output, _ = net(data.view(batch_size, -1))
					output, _ = net(data)
					_, idx = output.sum(dim=0).max(1)
					acc = np.mean((targets == idx).detach().cpu().numpy())

					print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")
					print('\n')

				counter += 1
				iter_counter +=1


	all_test_spk, all_test_mem = [], []
	all_test_data, all_test_targets = [], []
	with torch.no_grad():
		net.eval()
		for test_data, test_targets in iter(test_loader):
		
			# test_data, test_targets = next(iter(test_loader))
			test_data = test_data.to(device)
			test_targets = test_targets.to(device)

			# Test set forward pass
			test_spk, test_mem = net.fwd_frozen(test_data)
			all_test_spk.append(test_spk)
			all_test_mem.append(test_mem)
			all_test_data.append(test_data)
			all_test_targets.append(test_targets)


	visual_inputs = np.zeros([
		len(all_test_spk)*np.size(all_test_spk[0],1),
		28,
		28
	])
	visual_labels = np.zeros(len(all_test_spk)*np.size(all_test_spk[0],1))
	for i in range(len(all_test_spk)):
		visinput = all_test_data[i]
		vistarg = all_test_targets[i]
		ind_start = i*np.size(all_test_spk[0],1)
		ind_end = i*np.size(all_test_spk[0],1) + np.size(all_test_spk[0],1)
		visual_inputs[ind_start:ind_end, :, :] = np.squeeze(visinput).numpy()
		visual_labels[ind_start:ind_end] = vistarg.numpy()


	spike_outputs = np.zeros([
		2,
		81,
		len(all_test_spk)*np.size(all_test_spk[0],1),
		20
	])
	for i in range(len(all_test_spk)):
		spk = all_test_spk[i]
		mem = all_test_mem[i]
		ind_start = i*np.size(all_test_spk[0],1)
		ind_end = i*np.size(all_test_spk[0],1) + np.size(all_test_spk[0],1)
		spike_outputs[0, :, ind_start:ind_end, :] = spk.numpy()
		spike_outputs[1, :, ind_start:ind_end, :] = mem.numpy()

	np.savez(
		savepath+'.npz',
		inputs=visual_inputs,
		labels=visual_labels,
		outputs=spike_outputs
	)

	torch.save(net.state_dict(), savepath+'.pt')


if __name__ == '__main__':
	train_test_visual()
