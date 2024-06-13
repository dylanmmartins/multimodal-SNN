#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt


def combine_noisy_inputs():

    for i in ['0', '0p1', '0p2', '0p5', '1p0']:
        for j in ['0', '0p1', '0p2', '0p5', '1p0']:

            visdata = np.load('visnet_v2_input_noise_{}.npz'.format(i))
            auddata = np.load('audnet_v2_input_noise_{}.npz'.format(j))

            vis_inputs = visdata['inputs']
            aud_inputs = auddata['inputs']

            vis_labels = visdata['labels']
            aud_labels = auddata['labels']

            vis_outputs = visdata['outputs']
            aud_outputs = auddata['outputs']

            paired_inputs = np.zeros([
            np.size(aud_labels, 0),
                40,
                81
            ])

            usable_visInds = np.arange(np.size(vis_inputs, 0))

            for ind in range(np.size(aud_inputs, 0)):

                target_label = aud_labels[ind]

                possible_pairings = np.argwhere(vis_labels == target_label).flatten()

                possible_pairings = [x for x in possible_pairings if x in usable_visInds]

                paired_ind = possible_pairings[0]

                usable_visInds = np.delete(usable_visInds, paired_ind)

                for t in range(81):
                    paired_inputs[ind,:,t] = np.concatenate([
                        vis_outputs[1,t,paired_ind,:],
                        aud_outputs[1,t,ind,:]
                    ])

            np.save('paired_input_v{}_a{}.npy'.format(i,j), paired_inputs)
            np.save('paired_labels_v{}_a{}.npy'.format(i,j), aud_labels)


    vis_inputs = visdata['inputs']
    aud_inputs = auddata['inputs']

    vis_labels = visdata['labels']
    aud_labels = auddata['labels']

    vis_outputs = visdata['outputs']
    aud_outputs = auddata['outputs']

    paired_inputs = np.zeros([
        np.size(aud_labels, 0),
        40,
        81
    ])

    usable_visInds = np.arange(np.size(vis_inputs, 0))

    for ind in range(np.size(aud_inputs, 0)):

        target_label = aud_labels[ind]

        possible_pairings = np.argwhere(vis_labels == target_label).flatten()

        possible_pairings = [x for x in possible_pairings if x in usable_visInds]

        paired_ind = possible_pairings[0]

        usable_visInds = np.delete(usable_visInds, paired_ind)

        for t in range(81):
            paired_inputs[ind,:,t] = np.concatenate([
                vis_outputs[1,t,paired_ind,:],
                aud_outputs[1,t,ind,:]
            ])

    np.save('paired_hidden_noise_1p0.npy', paired_inputs)

    all_labels = []
    for f in [
        'audnet_v2_input_noise_0.npz',
        'audnet_v2_input_noise_0p1.npz',
        'audnet_v2_input_noise_0p2.npz',
        'audnet_v2_input_noise_0p5.npz',
        'audnet_v2_input_noise_1p0.npz',
        'audnet_v2_hidden_noise_0p1.npz',
        'audnet_v2_hidden_noise_0p2.npz',
        'audnet_v2_hidden_noise_0p5.npz',
        'audnet_v2_hidden_noise_1p0.npz'
    ]:
        labels = np.load(f)['labels']
        all_labels.append(labels)

    all_labels = np.stack(all_labels)

    np.save('all_labels.npy', all_labels)


if __name__ == '__main__':
    combine_noisy_inputs()