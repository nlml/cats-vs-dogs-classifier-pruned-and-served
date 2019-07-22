import os
from download_url import download_and_unzip_from_url, sizeof_fmt
import torch
from pathlib import Path
import numpy as np
from data_utils import split_dataset, get_torch_datasets, get_torch_loaders, get_transforms
from torchvision import utils
import torch.nn as nn
import nn_utils
from sgd_l1 import SGDL1
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from prune import prune_squeezenet


# Params #
DATA_DIR = '../data'
CHECKPOINTS_DIR = '../checkpoints'
DATASET_URL = 'https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip'  # noqa
SEED = 1
P_TRAIN = 0.7
P_VAL = 0.2
P_TEST = 0.1
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 100
NUM_PRUNING_ROUNDS = 10
NUM_EPOCHS_PER_PRUNING_ROUND = 10
PRUNING_FACTOR_PER_ROUND = 0.92
USE_CUDA = True
############


nn_utils.seed_everything(SEED)
device = 'cuda' if torch.cuda.is_available() and USE_CUDA else 'cpu'

data_dir = Path(DATA_DIR)
if not data_dir.exists():
    data_dir.mkdir(parents=True, exist_ok=True)
    download_and_unzip_from_url(DATASET_URL, data_dir)

if not Path(CHECKPOINTS_DIR).exists():
    Path(CHECKPOINTS_DIR).mkdir(parents=True, exist_ok=True)


dataset_dir = data_dir / 'PetImages'
rng = np.random.RandomState(SEED)

imagepaths, classes = split_dataset(
    dataset_dir, rng, P_TRAIN, P_VAL, P_TEST)

transform = get_transforms()

dataset = get_torch_datasets(transform, imagepaths, data_dir)


batch_size = {
    'train': TRAIN_BATCH_SIZE,
    'test': TEST_BATCH_SIZE,
    'val': TEST_BATCH_SIZE
}

loader = get_torch_loaders(dataset, batch_size)

phase_1_savename = 'squeezenet_post_output_layer_training.pth'
phase_2_savename = 'squeezenet_post_finetuning.pth'

phase_1_savepath = Path(CHECKPOINTS_DIR) / phase_1_savename
model = nn_utils.pretrained_squeezenet_with_two_output_classes()
if not (phase_1_savepath).exists():
    model = model.to(device)
    # Train just the new output layer for a few epochs w/ large learning rate
    optimizer = torch.optim.SGD(
        model.classifier.parameters(), 0.1, momentum=0.9)
    criterion = nn.NLLLoss()
    val_acc_history = nn_utils.train_model(
        device, model, loader, criterion, optimizer,
        num_epochs=3, postproc=lambda x: F.log_softmax(x, dim=1),
    )
    model = model.cpu()
    torch.save(model.state_dict(), phase_1_savepath)
    print(f'Saved to {phase_1_savepath}')
else:
    model.load_state_dict(torch.load(phase_1_savepath))

# Save the trace pre-model pruning for size comparison
model = model.cpu().eval()
trace = torch.jit.trace(model, torch.ones(1, 3, 224, 224))
fname = Path(CHECKPOINTS_DIR) / 'trace_pre_pruning.pth'
torch.jit.save(trace, str(fname))
print(f'Saved to {fname} - Size: ' + sizeof_fmt(fname.stat().st_size))


for i_pruning_round in range(NUM_PRUNING_ROUNDS):
    model = model.to(device)
    # Finetune the entire model on our cats/dogs dataset. Use a bit of L1 regularisation
    # to encourage weights to go to 0, which should help with pruning.
    l1_reg = 0.0001 * (NUM_PRUNING_ROUNDS - i_pruning_round) / NUM_PRUNING_ROUNDS
    optimizer = SGDL1(model.parameters(), 0.001, momentum=0.9, weight_decay=l1_reg)
    criterion = nn.NLLLoss()
    scheduler = ReduceLROnPlateau(
        optimizer, factor=0.5, patience=2, cooldown=0, verbose=True)
    first_or_last_round = (
        i_pruning_round == 0 or i_pruning_round == (NUM_PRUNING_ROUNDS - 1))
    val_acc_history = nn_utils.train_model(
        device, model, loader, criterion, optimizer,
        num_epochs=30 if first_or_last_round else NUM_EPOCHS_PER_PRUNING_ROUND,
        postproc=lambda x: F.log_softmax(x, dim=1),
        scheduler=scheduler
    )
    model = model.cpu()
    savename_this = (
        Path(CHECKPOINTS_DIR) /
        phase_2_savename.replace('.pth', ('_%02d.pth' % i_pruning_round))
    )
    torch.save(model.state_dict(), savename_this)
    print(f'Saved to {savename_this}')

    # Save the pruned + trained model
    model = model.cpu().eval()
    trace = torch.jit.trace(model, torch.ones(1, 3, 224, 224))
    fname = f'trace_pruned_{PRUNING_FACTOR_PER_ROUND ** i_pruning_round:.3f}.pth'
    fname = Path(CHECKPOINTS_DIR) / fname
    torch.jit.save(trace, str(fname))
    print(f'Saved to {fname} - Size: ' + sizeof_fmt(fname.stat().st_size))

    if i_pruning_round < NUM_PRUNING_ROUNDS - 1:
        # Prune out 5% of weights
        model = prune_squeezenet(model, PRUNING_FACTOR_PER_ROUND).to(device)
        print('Metrics after pruning:')
        eval_model(device, model, loader, criterion,
                   postproc=lambda x: F.log_softmax(x, dim=1))

copyfile(fname, Path('../final_pruned_model.pth'))
