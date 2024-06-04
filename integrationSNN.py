import numpy as np
from visualSNN import VisNet
import torch
from torch import nn
import snntorch as snn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from visualSNN import train_test_visual
from torch.optim import Adam

class MultimodalIntegration(nn.Module):
	"""
	Combine the outputs from the visual and auditory encoders into a single multimodal representation.
	"""
	def __init__(self, multimodal_size, output_size, beta):
		super(MultimodalIntegration, self).__init__()
		self.fc = nn.Linear(multimodal_size, output_size)
		self.lif = snn.Leaky(beta=beta)

	def forward(self, x):
		cur = self.fc(x)
		spk, mem = self.lif(cur)
		return spk, mem


def main():

	hidden_size = 50  # Number of neurons in the hidden layers
	beta = 0.95  # Decay rate of the LIF neuron
	multimodal_size = hidden_size * 2  # Combined size for multimodal layer

	# Output size for the final multimodal representation
	output_size = 10

	batch_size = 512
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

	state_dict1 = torch.load('vis_v1.pt')
	vismodel = VisNet(28*28).to(device)
	vismodel.load_state_dict(state_dict1)

	state_dict2 = torch.load('aud_v1.pt')
	audmodel = AudNet().to(device)
	audmodel.load_state_dict(state_dict2)

	intmodel = MultimodalIntegration(multimodal_size, output_size, beta).to_device()

	# Define loss function and optimizer
	loss_fn = nn.MSELoss() optimizer = Adam(net.parameters(), lr=0.01)

	# Training data is output from visual and audio encoders
	vis_spikes = torch.from_numpy(vis_spks).float()
	aud_spikes = torch.from_numpy(vis_spks).float() # audio doesn't work so just copy vision
	targets = torch.from_numpy(vis_labels).long() # (9984,)

	# one-hot encode targets
	#one_hot_targets = torch.zeros(9984, 10)
	#for i in range(len(targets)):
		#one_hot_targets[i, targets[i]] = 1
