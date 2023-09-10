import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import sys

import matplotlib.pyplot as plt
import IPython.display as ipd

from tqdm import tqdm
from data_loader import SubsetSC
from function import count_parameters, get_likely_index, number_of_correct, downsample, pad_sequence
from model import M5


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


def predict(tensor, sample_rate, new_sample_rate):
    # Use the model to predict the label of the waveform
    tensor = tensor.to(device)
    tensor = downsample(tensor, sample_rate, new_sample_rate)
    tensor = model(tensor.unsqueeze(0))
    tensor = get_likely_index(tensor)
    tensor = index_to_label(tensor.squeeze())
    return tensor


def test(model, epoch, sample_rate, new_sample_rate):
    model.eval()
    correct = 0
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)

        # apply transform and model on whole batch directly on device
        data = downsample(data, sample_rate, new_sample_rate)
        output = model(data)

        pred = get_likely_index(output)
        correct += number_of_correct(pred, target)

        # update progress bar
        pbar.update(pbar_update)

    print(
        f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n")


def train(model, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        data = data.to(device)
        target = target.to(device)

        # apply transform and model on whole batch directly on device
        data = downsample(data, sample_rate, new_sample_rate)
        output = model(data)

        # negative log-likelihood for a tensor of size (batch x 1 x n_output)
        loss = F.nll_loss(output.squeeze(), target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print training stats
        if batch_idx % log_interval == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

        # update progress bar
        pbar.update(pbar_update)
        # record loss
        losses.append(loss.item())


if __name__ == '__main__':
    global sample_rate
    global new_sample_rate

    new_sample_rate = 8000

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    batch_size = 256

    if device == "cuda":
        num_workers = 1
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False

    train_set = SubsetSC("training")
    test_set = SubsetSC("testing")

    waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]
    labels = sorted(list(set(datapoint[2] for datapoint in train_set)))
    # sorted all labels

    transformed_waveform = downsample(waveform, sample_rate, new_sample_rate)
    # transform data through downdampling function

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = M5(n_input=transformed_waveform.shape[0], n_output=len(labels))
    model.to(device)

    # optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    log_interval = 20
    n_epoch = 5

    pbar_update = 1 / (len(train_loader) + len(test_loader))
    losses = []

    # The transform needs to live on the same device as the model and the data.
    transform = transformed_waveform.to(device)
    with tqdm(total=n_epoch) as pbar:
        for epoch in range(1, n_epoch + 1):
            train(model, epoch, log_interval)
            torch.save({
                'epoch': n_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, "./ckpt/SGD_ckpt_epoch{}.pt".format(epoch))
            # save the ckpt
            test(model, epoch, sample_rate, new_sample_rate)
            scheduler.step()

    waveform, sample_rate, utterance, *_ = train_set[-1]

    print(f"Expected: {utterance}. Predicted: {predict(waveform, sample_rate, new_sample_rate)}.")

    for i, (waveform, sample_rate, utterance, *_) in enumerate(test_set):
        output = predict(waveform, sample_rate, new_sample_rate)
        if output != utterance:
            ipd.Audio(waveform.numpy(), rate=sample_rate)
            print(f"Data point #{i}. Expected: {utterance}. Predicted: {output}.")
            break
    else:
        print("All examples in this dataset were correctly classified!")
        print("In this case, let's just look at the last data point")
        ipd.Audio(waveform.numpy(), rate=sample_rate)
        print(f"Data point #{i}. Expected: {utterance}. Predicted: {output}.")

    # n = count_parameters(model)
    # print("Number of parameters: %s" % n)
