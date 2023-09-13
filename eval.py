import torch
import torchaudio

from model import M5
from data_loader import SubsetSC
from train import collate_fn
from function import downsample, get_likely_index, number_of_correct, pad_sequence

def collate_fn(batch):
    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number
    tensors, targets = [], []

    # Gather in lists, and encode labels as indices
    for waveform, _, label, *_ in batch:
        tensors += [waveform]
        targets += [label_to_index(label)]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    return tensors, targets

def label_to_index(word):
    # Return the position of the word in labels
    return torch.tensor(labels.index(word))


def index_to_label(index):
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return labels[index]

if __name__ == '__main__':
    global sample_rate
    global new_sample_rate
    optim_name = "SGD"
    sample_rate = 16000
    new_sample_rate = 8000
    batch_size = 256
    select_epoch = 35
    correct = 0

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    if device == "cuda":
        num_workers = 1
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False

    test_set = SubsetSC("testing")
    labels = sorted(list(set(datapoint[2] for datapoint in test_set)))
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = M5(n_input=len(test_set[0][0]), n_output=len(labels))
    weight = torch.load('/home/jysuh/PycharmProjects/Speech_command/ckpt/{}_ckpt_epoch{}.pth'.format(optim_name, select_epoch), map_location=device)
    model.load_state_dict(weight)

    with torch.no_grad():
        model.eval()
        data, sample_rate = torchaudio.load("/home/jysuh/Downloads/output.wav")

        # apply transform and model on whole batch directly on device
        data = downsample(data, sample_rate, new_sample_rate)
        output = model(data.unsqueeze(0))

        pred = get_likely_index(output)
        print(labels[int(pred)])