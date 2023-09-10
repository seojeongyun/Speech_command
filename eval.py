import torch

from model import M5
from data_loader import SubsetSC
from train import collate_fn
from function import downsample, get_likely_index, number_of_correct

if __name__ == '__main__':
    global sample_rate
    global new_sample_rate

    new_sample_rate = 8000
    batch_size = 256
    select_epoch = 5
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

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = M5(n_input=len(test_set[0][0]), n_output=35)
    weight = torch.load('./ckpt/SGD_ckpt_{}.pt'.format(select_epoch), map_location=device)
    model.load_state_dict(weight)

    with torch.no_grad():
        model.eval()
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)

            # apply transform and model on whole batch directly on device
            data = downsample(data, sample_rate, new_sample_rate)
            output = model(data)

            pred = get_likely_index(output)
            correct += number_of_correct(pred, target)
    print(f"\nAccuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n")