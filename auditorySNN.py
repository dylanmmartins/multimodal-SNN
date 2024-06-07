#import dependences 
import snntorch as snn
import torch
from torchvision import datasets, transforms
from snntorch import utils

from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import matplotlib.pyplot as plt
from snntorch import spikegen
import numpy as np

dtype = torch.float
torch.set_default_dtype(dtype)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


# Define Network
class AudNet(nn.Module):
	#no inputs for init,, 1000 hidden layers, 10 outputs, 81 steps, frequency 129, 
	def __init__(self,
		num_inputs=None, num_hidden=1000, num_last_hidden=20, num_output=10,
		beta=0.95, num_steps=81, num_freq=129):
		super().__init__()

		if num_inputs is None:
			num_inputs = num_steps * num_freq #number of inputs will be set by multiplying steps by frequency, or 129*81 if number of inputs is 0

		self.num_steps = num_steps
		self.num_freq = num_freq

		# Initialize layers 
		self.fc1 = nn.Linear(num_inputs, num_hidden) #first layer with num_inputs inputs and 1000 hidden 
		self.lif1 = snn.Leaky(beta=beta) #first layer of spiking neuro network, controlled by beta 
		self.fc2 = nn.Linear(num_hidden, num_hidden)
		self.lif2 = snn.Leaky(beta=beta)
		self.fc3 = nn.Linear(num_hidden, num_last_hidden)
		self.lif3 = snn.Leaky(beta=beta)
		self.fc4 = nn.Linear(num_last_hidden, num_output)
		self.lif4 = snn.Leaky(beta=beta)


	def forward(self, x):

		x.to(torch.float32) #converts x to a float32 
		n = x.shape[0] #sets n to the first dimension of x

		# Initialize hidden states at t=0
		mem1 = self.lif1.init_leaky()
		mem2 = self.lif2.init_leaky()
		mem3 = self.lif3.init_leaky()
		mem4 = self.lif3.init_leaky() #why lif3 and not lif4

		output_spike_record = []
		output_memV_record = []

		for step in range(self.num_steps):

			x_ = x.reshape(n, self.num_freq, self.num_steps)[:,:,step] #sets x_ to be the same dimensions as the original data 
			#print(np.shape(x_))
			#x_ = x.reshape(self.num_freq, self.num_steps)[:,:,step]
			cur1 = self.fc1(x_) 
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
		n = x.shape[0]
        
		mem1 = self.mem1
		mem2 = self.mem2
		mem3 = self.mem3

		last_hidden_spike_record = []
		last_hidden_output_memV_record = []


		for step in range(self.num_steps):
			#print(np.shape(x))
			#x_ = x.reshape(self.num_freq, self.num_steps)[:,step]
			#x_ = x.reshape(self.num_freq, self.num_steps)[:,:,step]
			#print(np.shape(x_))

			x_ = x.reshape(n, self.num_freq, self.num_steps)[:, :, step]
   
			#x_ = x_.reshape(n, -1)

			cur1 = self.fc1(x_)
			spk1, mem1 = self.lif1(cur1, mem1)

			cur2 = self.fc2(spk1)
			spk2, mem2 = self.lif2(cur2, mem2)

			cur3 = self.fc3(spk2)
			spk3, mem3 = self.lif3(cur3, mem3)

			last_hidden_spike_record.append(spk3)
			last_hidden_output_memV_record.append(mem3)

		return torch.stack(last_hidden_spike_record, dim=0), torch.stack(last_hidden_output_memV_record, dim=0)


def train_test_auditory(savepath=None):
	
	audio_stim = np.load('auditory_stimuli.npz') #load the mpz file as the auditory stimulus the keys are x_train, y_train, x_test, y_test
	X_train = audio_stim['X_train'] #size of (2100, 129, 81) number of samples x frequency x time 
	X_test = audio_stim['X_test'] #size of (900, 129, 81)
	y_train = audio_stim['y_train'] #size (2100 by 1)
	y_test = audio_stim['y_test'] #size of 900 by 1

	# Dataloader arguments
	batch_size = 128*4

	# Define a transform
	transform = transforms.Compose([
		transforms.ToTensor()
	])

	training_dataset = TensorDataset(
		torch.from_numpy(X_train.astype(np.float32)),
		torch.from_numpy(y_train.astype(np.float32))
	) # creates a tensor of float32 containing the training data and label 

	testing_dataset = TensorDataset(
		torch.from_numpy(X_test.astype(np.float32)),
		torch.from_numpy(y_test.astype(np.float32))
	) #creates a dataset for testing containihng data and label in a float32 format

	# Create DataLoaders
	train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, drop_last=True) 
	test_loader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

	num_epochs = 100 #control the number of epochs in the training
	loss_hist = []
	test_loss_hist = []
	counter = 0

	input_sz = np.size(X_train, 1) #sets input size to  the second dimension of x_train. For us 129
	# Load the network onto CUDA if available
	net = AudNet(num_inputs=input_sz).to(device) #loads the data using 129 as the number of inputs
	#print("network loaded")

	# pass data into the network, sum the spikes over time
	# and compare the neuron with the highest number of spikes
	# with the target

	loss = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))

	# Outer training loop
	for epoch in range(num_epochs):
		iter_counter = 0
		train_batch = iter(train_loader)
		#print(epoch)
		# Minibatch training loop
		for data, targets in train_batch:
			data = data.to(device)
			targets = targets.to(device)
			#print(data)
			#print(targets)
			# forward pass
			net.train()\

   
			#spk_rec, mem_rec = net(data.view(batch_size, -1)) #batch x freq * freq*num_steps x hidden 
			spk_rec, mem_rec = net(data)
			#spk_rec, mem_rec  = net.fwd_frozen(data)
			#print("spk_rec, mem_rec")
			# initialize the loss & sum over time
			# loss_val = torch.zeros((1), dtype=dtype, device=device)
			loss_val = loss(mem_rec[-1], targets.type(torch.long))

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
#				test_spk, test_mem = net(test_data.view(batch_size, -1))
#				test_spk, test_mem  = net.fwd_frozen(test_data)
				test_spk, test_mem = net(test_data)
#				test_spk, test_mem = net.fwd_frozen(test_data)
				# Test set loss
				# test_loss = torch.zeros((1), dtype=dtype, device=device)
				# for step in range(num_steps):
				test_loss = loss(test_mem[-1], test_targets.type(torch.long))
				test_loss_hist.append(test_loss.item())

				# Print train/test loss/accuracy
				if counter % 50 == 0:
					
					print(f"Epoch {epoch}, Iteration {iter_counter}, Train loss = {loss_hist[counter]:.2f} Test loss = {test_loss_hist[counter]:.2f} \n")

					#output, _ = net(data.view(batch_size, -1))
					audio_spikes,  audio_mem = net.fwd_frozen(data.view(batch_size, -1))
					_, idx = audio_mem.sum(dim=0).max(1)
					acc = np.mean((targets == idx).detach().cpu().numpy())
					print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")
					print('\n')
     
					torch.save(net.state_dict(), "aud_v1.pt")

					
				counter += 1
		
			iter_counter +=1

	#print(net.state_dict())
	#torch.save(net.state_dict(), savepath+'.pt')


if __name__ == '__main__':
	train_test_auditory('aud_v1')
 
 




output = []
output.append(audio_spikes)
output.append(audio_mem)
     
np.savez(

	'auditory_stim_outputs.npz',
	inputs = np.concatenate((X_train, X_test), axis=0),
	labels=np.concatenate((y_train, y_test), axis=0, out=None),
	outputs =output,
	)
