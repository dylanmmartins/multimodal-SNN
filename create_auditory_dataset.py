# write an .npz file of MNIST audio recordings. normalize, do train/test split, etc.
# spoken MNIST audio wave files from: https://github.com/Jakobovski/free-spoken-digit-dataset/tree/master

from scipy.io import wavfile
from glob import glob
import os
import numpy as np


def main(fpath):

    
    wavfiles = glob(fpath)
    
    # find files and org by digit. append to a list of numpy arrays, then
    # add that arr to a dict. keys are 0 to 9 as str
    audio_dict = {}
    for f in wavfiles:
        digit = os.path.split(f)[1][0]
        _, wavsample = wavfile.read(f)
        if digit not in audio_dict.keys():
            audio_dict[digit] = []
        audio_dict[digit].append(wavsample)
        
    # get max len of each recording
    # recordings are all diff lengths, so pad all fo the samples with zeros
    # once the recording is over. obv better than stretching them out, might
    # want to change this later, though...
    max_len = 0
    for k,v in audio_dict.items():
        for x in v:
            if len(x)>max_len:
                max_len = len(x)
    audio_stimuli = np.zeros([300*10, max_len])
    audio_labels = np.zeros([300*10])

    i = 0
    for k,v in audio_dict.items():
        for x in v:

            audio_labels[i] = float(k)

            _tmp_x = np.zeros(np.size(audio_stimuli,1))
            _tmp_x[:len(x)] = x.copy()
            audio_stimuli[i,:] = _tmp_x

            i += 1

    # normalize from 0 to 1
    norm_audio_stimuli = audio_stimuli + np.abs(np.min(audio_stimuli))
    norm_audio_stimuli = norm_audio_stimuli / np.max(norm_audio_stimuli)

    # train/test split. every digit has the same frac in train/test, but
    # it's randomly sampled within each digit
    split_frac = 0.7

    test_inds = []
    train_inds = []

    for si in np.arange(0,3000,300):

        all_inds = np.arange(300)
        rsamp = np.random.choice(all_inds, size=int(np.ceil(300*split_frac)), replace=False)
        train_ = rsamp + si
        test_ = np.array([x for x in all_inds if x not in rsamp]) + si
        test_inds.extend(list(test_))
        train_inds.extend(list(train_))

    train_inds = np.array(sorted(train_inds))
    test_inds = np.array(sorted(test_inds))

    X_train = norm_audio_stimuli[train_inds,:]
    X_test = norm_audio_stimuli[test_inds,:]
    y_train = audio_labels[train_inds]
    y_test = audio_labels[test_inds]

    np.savez('auditory_stimuli.npz',
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )



if __name__ == '__main__':

    data_directory = r'T:\mnist\free-spoken-digit-dataset-master\recordings\*.wav'
    main(data_directory)