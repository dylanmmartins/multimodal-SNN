
import snntorch as snn
import torch
from torchvision import datasets, transforms
from snntorch import utils
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from snntorch import spikegen
import numpy as np

dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Dataloader arguments
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

    def set_noise(self, noise_std=0, add_at='input'):

        self.noise_level = noise_std

        if add_at == 'input':
            self.noise_method = 0
        elif add_at == 'hidden':
            self.noise_method = 1

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

        mem1 = self.mem1
        mem2 = self.mem2
        mem3 = self.mem3

        last_hidden_spike_record = []
        last_hidden_output_memV_record = []

        if self.noise_method == 0:
            x = x + (self.noise_level * torch.randn_like(x))

        for step in range(self.num_steps):

            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            last_hidden_spike_record.append(spk3)
            last_hidden_output_memV_record.append(mem3)

        sp_out = torch.stack(last_hidden_spike_record, dim=0)
        mem_out = torch.stack(last_hidden_output_memV_record, dim=0)

        if self.noise_method == 1:
            sp_out = sp_out + (self.noise_level * torch.randn_like(sp_out))
            mem_out = mem_out + (self.noise_level * torch.randn_like(mem_out))

        return sp_out, mem_out
    
def train_visual(savename):

    data_path='/tmp/data/mnist'

    # Define a transform
    transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0,), (1,))])

    mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

    # Create DataLoaders
    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)


    # Network Architecture
    num_inputs = 28*28

    num_epochs = 3
    loss_hist = []
    test_loss_hist = []
    counter = 0

    # Load the network onto CUDA if available
    net = VisNet(num_inputs=28*28).to(device)
    #print(net.device)
    # next(net.parameters()).is_cuda

    net.set_noise(0, add_at=None)

    def print_batch_accuracy(data, targets, train=False):
        output, _ = net(data.view(batch_size, -1))
        _, idx = output.sum(dim=0).max(1)
        acc = np.mean((targets == idx).detach().cpu().numpy())

        # if train:
        #     print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
        # else:
        print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")

    def train_printer():
        print(f"Epoch {epoch}, Iteration {iter_counter}, Train loss = {loss_hist[counter]:.2f} Test loss = {test_loss_hist[counter]:.2f} \n")
        print_batch_accuracy(data, targets)
        print('\n')

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
                test_spk, test_mem = net(test_data.view(batch_size, -1))

                # Test set loss
                test_loss = torch.zeros((1), dtype=dtype, device=device)
                for step in range(net.num_steps):
                    test_loss += loss(test_mem[step], test_targets)
                test_loss_hist.append(test_loss.item())

                # Print train/test loss/accuracy
                if counter % 50 == 0:
                    train_printer()
                counter += 1
                iter_counter +=1

    torch.save(net.state_dict(), savename+'.pt')

    return net, train_loader, test_loader

def test_visual(savename, net, train_loader, test_loader, noise_method='input', noise_std=0):

    net.set_noise(noise_std, add_at=noise_method)

    all_test_spk, all_test_mem = [], []
    all_test_data, all_test_targets = [], []

    with torch.no_grad():
        net.eval()
        for test_data, test_targets in iter(test_loader):
		
            test_data = test_data.to(device)
            test_targets = test_targets.to(device)

            # Test set forward pass
            test_spk, test_mem = net.fwd_frozen(test_data.view(batch_size, -1))
            all_test_spk.append(test_spk)
            all_test_mem.append(test_mem)
            all_test_data.append(test_data)
            all_test_targets.append(test_targets)

    visual_inputs = np.zeros([len(all_test_spk)*np.size(all_test_spk[0],1), 28, 28])
    visual_labels = np.zeros(len(all_test_spk)*np.size(all_test_spk[0],1))
    for i in range(len(all_test_spk)):
        visinput = all_test_data[i]
        vistarg = all_test_targets[i]
        ind_start = i*np.size(all_test_spk[0],1)
        ind_end = i*np.size(all_test_spk[0],1) + np.size(all_test_spk[0],1)
        visual_inputs[ind_start:ind_end, :, :] = np.squeeze(visinput).numpy()
        visual_labels[ind_start:ind_end] = vistarg.numpy()


    spike_outputs = np.zeros([2, 81, len(all_test_spk)*np.size(all_test_spk[0],1), 20])
    for i in range(len(all_test_spk)):
        spk = all_test_spk[i]
        mem = all_test_mem[i]
        ind_start = i*np.size(all_test_spk[0],1)
        ind_end = i*np.size(all_test_spk[0],1) + np.size(all_test_spk[0],1)
        spike_outputs[0, :, ind_start:ind_end, :] = spk.numpy()
        spike_outputs[1, :, ind_start:ind_end, :] = mem.numpy()

    np.savez(savename+'.npz', inputs=visual_inputs, labels=visual_labels, outputs=spike_outputs)


if __name__ == '__main__':

    net, train_loader, test_loader = train_visual('visnet_v2')

    noise_method = 'hidden'
    noise_levels = [0., 0.1, 0.2, 0.5, 1.0]
    noise_names = ['0', '0p1', '0p2', '0p5', '1p0']

    noise_levels = noise_levels[1:]
    noise_names = noise_names[1:]

    for i in range(len(noise_levels)):

        nl = noise_levels[i]
        nname = noise_names[i]

        print('Testing VisNet with {} noise with std of {}'.format(noise_method, nl))

        test_visual('visnet_v2_{}_noise_{}'.format(noise_method, nname),
                    net, train_loader, test_loader, noise_method=noise_method, noise_std=nl)
