import os
import torch
import matplotlib.pyplot as plt
import IPython.display as ipd
import torchaudio

from function import pad_sequence
from torchaudio.datasets import SPEECHCOMMANDS



class SubsetSC(SPEECHCOMMANDS): # Data loader
    def __init__(self, subset: str = None):
        super().__init__("/storage/jysuh/", download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]

if __name__ == '__main__':
    # Create training and testing split of the data. We do not use validation in this tutorial.
    train_set = SubsetSC("training")
    test_set = SubsetSC("testing")

    waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]

    print("Shape of waveform: {}".format(waveform.size()))
    print("Sample rate of waveform: {}".format(sample_rate))

    labels = sorted(list(set(datapoint[2] for datapoint in train_set)))
    # The name of classe is sorted.
    # backward, bed, bird, cat, .... , wow, yes, zero

    waveform_first, *_ = train_set[0]
    ipd.Audio(waveform_first.numpy(), rate=sample_rate)

    # waveform_second, *_ = train_set[1]
    # ipd.Audio(waveform_second.numpy(), rate=sample_rate)

    plt.plot(waveform.t().numpy())
    plt.show()
