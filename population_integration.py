# %% [markdown]
# ## Imports

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
        print(f"Shape after flattening: {x.shape}")
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

# ## Hyperparameters
beta = 0.9  # Decay rate of the LIF neuron
hidden_size = 128  # Size of the hidden layer
output_size = 10  # Output size for the final multimodal representation
input_size = 40  # Correct input size based on concatenation of visual and auditory spikes

# Load the visual outputs file
visual_npzfile = np.load("visual_outputs.npz")
print("Visual Outputs Keys:", visual_npzfile.keys())

# Load the auditory outputs file
audio_npzfile = np.load("auditory_stim_outputs.npz")
print("Audio Outputs Keys:", audio_npzfile.keys())

"""
Visual Outputs Keys:
'inputs', 'labels', 'outputs'

Auditory Outputs Keys:
'inputs', 'labels', 'spikes_output', 'mem_output'
"""

# Visual data
vis_labels = visual_npzfile["labels"]  # (9984,)
vis_outputs = visual_npzfile["outputs"]  # (2, 25, 9984, 50)
vis_spks = np.transpose(vis_outputs[0, :, :, :], (1, 0, 2))  # (9984, 25, 50)

# Auditory data
aud_labels = audio_npzfile["labels"]  # (93000,)
aud_spks_raw = audio_npzfile["spikes_output"]
aud_spks = np.transpose(aud_spks_raw, (1, 0, 2))  # (93000, 25, 50)

vis_rep = torch.from_numpy(vis_spks[0]).float()
aud_rep = torch.from_numpy(aud_spks[0])

# Concatenate the encoded spikes
combined_spikes = torch.cat((vis_rep, aud_rep), dim=1)  # (25, 100)
combined_spikes = combined_spikes.permute(1, 0)  # (100, 25)
print(f"Shape of combined_spikes: {combined_spikes.shape}")

net = MultimodalIntegration(input_size, hidden_size, output_size, beta)

# Forward pass through the integration layer
# multimodal_output, mem = net(combined_spikes)
# print("Multimodal output spike vector:", multimodal_output)

loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(net.parameters(), lr=0.01)

# Training data is output from visual and audio encoders
vis_spikes = torch.from_numpy(vis_spks).float()
aud_spikes = torch.from_numpy(aud_spks).float()
targets = torch.from_numpy(vis_labels).long()  # (9984,)

# Training loop
num_epochs = 10
num_samples = min(vis_spikes.shape[0], aud_spikes.shape[0], targets.shape[0])  # Use the minimum size across datasets
num_steps = 25  # Number of timesteps for the spike train

for epoch in range(num_epochs):
    epoch_loss = 0.0
    for i in range(num_samples):
        target = targets[i].unsqueeze(0)

        # Concatenate output spike trains from single sensory encoders
        spike_train = torch.cat((vis_spikes[i, :, :], aud_spikes[i, :, :]), dim=1)  # (25, 100)
        print(f"Shape of spike_train: {spike_train.shape}")

        # Initialize hidden states
        mem1 = net.lif1.init_leaky()
        mem2 = net.lif2.init_leaky()
        mem3 = net.lif3.init_leaky()

        # Accumulate loss over timesteps
        total_loss = 0.0
        optimizer.zero_grad()
        for t in range(num_steps):
            # Flatten the spike train for the current timestep
            input_t = spike_train[t].view(1, -1)  # (1, 100)
            print(f"Shape of input_t at timestep {t}: {input_t.shape}")

            # Forward pass through the network for each timestep
            spk3, mem3 = net(input_t)

            # Compute loss for this timestep
            loss = loss_fn(spk3, target)
            total_loss += loss

        # Backward pass and optimization
        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.item()

    avg_loss = epoch_loss / num_samples
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}')

print("Training complete")

# %% [markdown]
# ## Inference

# %% [markdown]
# ### Plotting code

def do_and_plot_model_inference(visual_spikes, audio_spikes, targets, ex=0, hidden_size=81, audio_type="random"):
    """
    audio_type: "random", "visual_copy"
    """
    visual_spk = visual_spikes[ex, :, :]

    if audio_type == "visual_copy":
        audio_spk = visual_spk
    elif audio_type == "random":
        audio_input = torch.rand((40, 81))
        audio_spk = spikegen.rate(audio_input, num_steps=25)
    else:
        print("Not an option...")
        return

    target = targets[ex].unsqueeze(0)

    # Concatenate the encoded spikes
    combined_spikes = torch.cat((visual_spk, audio_spk), dim=1)  # (25, 100)

    # Forward pass through the integration layer
    outputs = []
    mem3 = net.lif3.init_leaky()
    for t in range(num_steps):
        input_t = combined_spikes[t].view(1, -1)
        output, mem3 = net(input_t)
        outputs.append(output)
    outputs = torch.stack(outputs)

    print(f"Vision Input: shape={visual_spk.shape}, spikes={visual_spk.sum()}")
    print(f"Audio: shape={audio_spk.shape}, spikes={audio_spk.sum()}")
    print(f"Output: shape={outputs.shape}, spikes={outputs.sum()}")
    print(f"Ground truth: {target}")

    fig, ax = plt.subplots(3, figsize=(8, 7), sharex=True,
                           gridspec_kw={'height_ratios': [1, 1, 0.4]})

    # Plot hidden layer spikes
    splt.raster(visual_spk, ax[0], s=5, c="black")
    ax[0].set_ylabel("Vision Hidden")

    # Plot hidden layer spikes
    splt.raster(audio_spk, ax[1], s=5, c="black")
    ax[1].set_ylabel("Audio Hidden")

    # Plot output spikes
    splt.raster(outputs.reshape(num_steps, -1), ax[2], c="black", marker="|")
    ax[2].set_ylabel("Output")

    ax[0].set_ylim((-2.45, 51.45))
    ax[1].set_ylim((-2.45, 51.45))
    ax[2].set_ylim([0, 9])

    plt.tight_layout()
    plt.show()

# %% [markdown]
# ### Visual and audio the same

do_and_plot_model_inference(vis_spikes, aud_spikes, targets, ex=1, audio_type="visual_copy")

# %% [markdown]
# ### Vision good but audio random

do_and_plot_model_inference(vis_spikes, aud_spikes, targets, ex=1, audio_type="random")

# %% [markdown]
# # Questions and todo
# 
# * include audio hidden output
# * how to convert multimodal output spike train into final answer
