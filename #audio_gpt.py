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
    def __init__(self, num_inputs=None, num_hidden=1000, num_last_hidden=20, num_output=10, beta=0.95, num_steps=81, num_freq=129):
        super().__init__()

        if num_inputs is None:
            num_inputs = num_steps * num_freq

        self.num_steps = num_steps
        self.num_freq = num_freq

        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.lif2 = snn.Leaky(beta=beta)
        self.fc3 = nn.Linear(num_hidden, num_last_hidden)
        self.lif3 = snn.Leaky(beta=beta)
        self.fc4 = nn.Linear(num_last_hidden, num_output)
        self.lif4 = snn.Leaky(beta=beta)

    def forward(self, x):
        x = x.to(torch.float32)
        n = x.shape[0]

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()

        output_spike_record = []
        output_memV_record = []

        for step in range(self.num_steps):
            x_ = x.reshape(n, self.num_freq, self.num_steps)[:, :, step]
            x_ = x_.reshape(n, -1)
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

        return torch.stack(output_spike_record, dim=0), torch.stack(output_memV_record, dim=0)

    def fwd_frozen(self, x):
        x = x.to(torch.float32)
        n = x.shape[0]

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()

        output_spike_record = []
        output_memV_record = []

        for step in range(self.num_steps):
            x_ = x.reshape(n, self.num_freq, self.num_steps)[:, :, step]
            x_ = x_.reshape(n, -1)
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

        return torch.stack(output_spike_record, dim=0), torch.stack(output_memV_record, dim=0)

if __name__ == '__main__':
    train_test_auditory('aud_v1')
