#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import snntorch as snn
import snntorch.functional as SF
from snntorch import spikeplot as splt
import snntorch.spikegen as spikegen
import torch
from torch import nn
from torch.optim import Adam

class MultimodalIntegration(nn.Module):
	"""
	Combine the outputs from the visual and auditory encoders into a single multimodal representation.
	"""

	def __init__(self, input_size, hidden_size, output_size, beta):
		super(MultimodalIntegration, self).__init__()

		self.flatten = nn.Flatten()
		self.fc1 = nn.Linear(input_size, hidden_size)
		self.lif1 = snn.Leaky(beta=beta)
		self.fc2 = nn.Linear(hidden_size, hidden_size)
		self.lif2 = snn.Leaky(beta=beta)
		self.fc3 = nn.Linear(hidden_size, output_size)
		self.lif3 = snn.Leaky(beta=beta)

	def forward(self, x):
		x = self.flatten(x.to(torch.float32))
		#print(f"Shape after flattening: {x.shape}")
		mem1 = self.lif1.init_leaky()
		mem2 = self.lif2.init_leaky()
		mem3 = self.lif3.init_leaky()
		cur1 = self.fc1(x)
		spk1, mem1 = self.lif1(cur1, mem1)
		cur2 = self.fc2(spk1)
		spk2, mem2 = self.lif2(cur2, mem2)
		cur3 = self.fc3(spk2)
		spk3, mem3 = self.lif3(cur3, mem3)
		return spk3, mem3


def train_integration(sensory_data_path, labels_path):

	beta = 0.9  # Decay rate of the LIF neuron
	hidden_size = 128  # Size of the hidden layer
	output_size = 10  # Output size for the final multimodal representation
	input_size = 40  # Correct input size based on concatenation of visual and auditory spikes

	sensory_data = np.load(sensory_data_path)
	labels = np.load(labels_path)

	combined_spikes = torch.from_numpy(sensory_data)

	net = MultimodalIntegration(input_size, hidden_size, output_size, beta)

	loss_fn = nn.CrossEntropyLoss()
	optimizer = Adam(net.parameters(), lr=0.01)

	targets = torch.from_numpy(labels)
	num_epochs = 10

	num_steps = 81

	for epoch in range(num_epochs):
		epoch_loss = 0.0

		for i in range(np.size(sensory_data, 0)):

			target = targets[i].unsqueeze(0)

			# Initialize hidden states
			mem1 = net.lif1.init_leaky()
			mem2 = net.lif2.init_leaky()
			mem3 = net.lif3.init_leaky()

			# Accumulate loss over timesteps
			total_loss = 0.0
			optimizer.zero_grad()

			for t in range(num_steps):
				# Flatten the spike train for the current timestep
				input_t = combined_spikes[i,:,t].unsqueeze(0)

				# Forward pass through the network for each timestep
				spk3, mem3 = net(input_t)

				# Compute loss for this timestep
				loss = loss_fn(spk3, target.long())
				total_loss += loss

			# Backward pass and optimization
			optimizer.zero_grad()
			total_loss.backward()
			optimizer.step()

			epoch_loss += total_loss.item()

		avg_loss = epoch_loss / np.size(sensory_data, 0)
		print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}')

		batch = 512
		correct_preds = 0
		false_preds = 0
		with torch.no_grad():
			net.eval()
			for i in range(np.size(sensory_data, 0)):
				target = targets[i].unsqueeze(0)
				for t in range(num_steps):
					input = combined_spikes[i,:,t].unsqueeze(0)
					spk, mem = net(input)
					if np.argmax(mem) == target:
						correct_preds += 1
					else:
						false_preds += 1

		print('Model using {} and {}'.format(sensory_data_path, labels_path))
		print(f'Correct predictions: {correct_preds}, False predictions: {false_preds}')

		params = [param for param in net.parameters()]
		l1_weights = np.abs(params[0].sum(0).detach().numpy())

	return l1_weights

def jitter(x, sz, scale):
	return x + np.random.uniform(-scale, scale, sz)

def population_integration():
	
	sensory_data = np.load('paired_input_noise_0p1.npy')
	all_labels = np.load('all_labels.npy')
	labels = all_labels[1,:].copy()

	beta = 0.9  # Decay rate of the LIF neuron
	hidden_size = 128  # Size of the hidden layer
	output_size = 10  # Output size for the final multimodal representation
	input_size = 40  # Correct input size based on concatenation of visual and auditory spikes

	combined_spikes = torch.from_numpy(sensory_data)

	net = MultimodalIntegration(input_size, hidden_size, output_size, beta)

	combined_spikes = torch.from_numpy(sensory_data)

	outputs = []
	memV = []

	ex = 5

	num_time_points = 81
	for t in range(num_time_points):
		input_t_rate = combined_spikes[ex,:,t]
		sp, mem = net(input_t_rate.unsqueeze(0))
		outputs.append(sp)
		memV.append(mem)

	outputs = torch.stack(outputs)
	memV = torch.stack(memV)

	fig, [ax1,ax2,ax3] = plt.subplots(3,1, dpi=300, figsize=(6.5,5))

	set_min = combined_spikes[ex].min().numpy()
	set_max = combined_spikes[ex].max().numpy()

	set_abs_max = np.max(np.abs([set_min, set_max]))
	set_min = -set_abs_max
	set_max = set_abs_max

	ax1.imshow(combined_spikes[ex,:20,:], vmin=set_min, vmax=set_max, cmap='coolwarm', aspect=0.8)
	ax2.imshow(combined_spikes[ex,20:,:], vmin=set_min, vmax=set_max, cmap='coolwarm', aspect=0.8)

	ax1.invert_yaxis()
	ax2.invert_yaxis()
	ax1.set_yticks(np.linspace(-0.5,19.5,5), labels=np.linspace(0,20,5).astype(int))
	ax2.set_yticks(np.linspace(-0.5,19.5,5), labels=np.linspace(0,20,5).astype(int))

	ax1.set_xticklabels([])
	ax2.set_xticklabels([])
	for i in range(10):
		ax3.vlines(np.argwhere(outputs.detach().numpy()[:,0,i]).flatten(), i-0.5, i+0.5, color='k')
	ax3.set_ylim([-0.5,9.5])
	ax3.set_yticks(np.arange(10), labels=np.arange(10).astype(int))

	ax1.set_ylabel('Visual Neurons')
	ax2.set_ylabel('Auditory Neurons')
	ax3.set_ylabel('Output Neurons')

	ax3.set_xlabel('time (a.u.)')
	ax3.set_xlim([0,81])
	ax3.set_title('Ground Truth = {}'.format(int(labels[ex])))

	fig.tight_layout()

	params = [param for param in net.parameters()]


	l1_weights = np.abs(params[0].sum(0).detach().numpy())

	fig, [ax1,ax2] = plt.subplots(1,2, dpi=300, figsize=(7,3))

	ax1.plot(l1_weights[:20], 'ko')
	ax1.set_xlim([-1,20])
	ax1.set_ylim([0,58])
	# ax1.set_title('visual mean={:.2f}'.format(l1_weights[:20].mean()))

	ax2.plot(l1_weights[20:], 'ko')
	ax2.set_xlim([-1,20])
	ax2.set_ylim([0,58])
	# ax2.set_title('auditory mean={:.2f}'.format(l1_weights[20:].mean()))

	ax1.set_ylabel('Weight')
	ax1.set_xlabel('Visual Neurons')

	ax2.set_xlabel('Auditory Neurons')
	ax2.set_ylabel('Weight')

	fig.tight_layout()


	fig, ax1 = plt.subplots(1,1, dpi=300, figsize=(2,3))
	ax1.plot(jitter(0, 20, 0.2), l1_weights[:20], 'k.')
	ax1.plot(jitter(1, 20, 0.2), l1_weights[20:], 'k.')
	ax1.set_xlim([-0.5,1.5])
	ax1.set_ylim([0,58])
	ax1.set_xticks([0,1], labels=['Visual','Auditory'])
	ax1.hlines(l1_weights[:20].mean(), -0.3, 0.3, color='tab:red')
	ax1.hlines(l1_weights[20:].mean(), 0.7, 1.3, color='tab:red')
	v_stderr = l1_weights[:20].std() / np.sqrt(20)
	a_stderr = l1_weights[20:].std() / np.sqrt(20)
	ax1.vlines(
		0,
		l1_weights[:20].mean()-v_stderr,
		l1_weights[:20].mean()+v_stderr,
		color='tab:red'
	)
	ax1.vlines(
		1,
		l1_weights[20:].mean()-a_stderr,
		l1_weights[20:].mean()+a_stderr,
		color='tab:red'
	)
	ax1.set_ylabel('Weight')
	fig.tight_layout()

	all_model_l1s = {}
	for i in ['0', '0p1', '0p2', '0p5', '1p0']:
		for j in ['0', '0p1', '0p2', '0p5', '1p0']:
			l1_weights = train_integration('paired_input_v{}_a{}.npy'.format(i,j), 'paired_labels_v{}_a{}.npy'.format(i,j))
			all_model_l1s['v{}_a{}'.format(i,j)] = l1_weights

	fig, ax = plt.subplots(3, figsize=(8, 7), sharex=True,
							gridspec_kw={'height_ratios': [1, 1, 0.4]})

	plt.plot(combined_spikes[ex,:20,:].T, ax[0], s=5, c="black", marker="|")

	ax[0].set_ylabel("Vision Hidden")

	splt.raster(combined_spikes[ex,20:,:].T, ax[1], s=5, c="black")
	ax[1].set_ylabel("Audio Hidden")

	splt.raster(outputs[:,0,:], ax[2], c="black", marker="|")
	ax[2].set_ylabel("Output")

	plt.tight_layout()
	plt.show()
