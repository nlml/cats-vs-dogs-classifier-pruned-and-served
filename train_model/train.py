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
import argparse
from shutil import copyfile


# Read in training parameters
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--data-dir', type=str, default='../data')
parser.add_argument('--checkpoints-dir', type=str, default='../checkpoints')
parser.add_argument('--dataset-url', type=str, default='https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip')  # noqa
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--p-train', type=float, default=0.7,
                    help='proportion of dataset to use as training set (default: 0.7)')
parser.add_argument('--p-val', type=float, default=0.2,
                    help='proportion of dataset to use as validation set (default: 0.2)')
parser.add_argument('--p-test', type=float, default=0.1,
                    help='proportion of dataset to use as test set (default: 0.1)')
parser.add_argument('--num-pruning-rounds', type=int, default=10,
                    help='how many pruning rounds (sets of epochs between model prunes) to run')
parser.add_argument('--num-epochs-per-pruning-round', type=int, default=10,
                    help='how many epochs to train for in each pruning round')
parser.add_argument('--pruning-factor-per-round', type=float, default=0.92,
                    help='Proportion of model weights remaining after a single pruning operation')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--learning-rate', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--l1-reg-weight', type=float, default=0.0001, metavar='L1',
                    help='Weight on L1 regularisation penalty')
args = parser.parse_args()

# Set torch random seed and device
nn_utils.seed_everything(args.seed)
device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'

# Download cats vs. dogs dataset if needed
data_dir = Path(args.data_dir)
if not (data_dir / 'PetImages').exists():
    data_dir.mkdir(parents=True, exist_ok=True)
    download_and_unzip_from_url(args.dataset_url, data_dir)
dataset_dir = data_dir / 'PetImages'

# Create checkpoints save dir if needed
if not Path(args.checkpoints_dir).exists():
    Path(args.checkpoints_dir).mkdir(parents=True, exist_ok=True)

# Randomly split the cats vs. dogs dataset into train/valid/test portions
rng = np.random.RandomState(args.seed)
imagepaths, classes = split_dataset(
    dataset_dir, rng, args.p_train, args.p_val, args.p_test)

# Get pytorch datasets and dataloaders
transform = get_transforms()
dataset = get_torch_datasets(transform, imagepaths, data_dir)
batch_size = {
    'train': args.batch_size,
    'test': args.test_batch_size,
    'val': args.test_batch_size
}
loader = get_torch_loaders(dataset, batch_size)

# Some filenames for intermediate checkpointing
phase_1_savename = 'squeezenet_post_output_layer_training.pth'
phase_2_savename = 'squeezenet_post_finetuning.pth'
phase_1_savepath = Path(args.checkpoints_dir) / phase_1_savename

# Load squeezenet pretrained on imagenet, with its
# output layer replaced to have just 2 outputs.
model = nn_utils.pretrained_squeezenet_with_two_output_classes()

# First we train just the new output layer for a few epochs w/ large learning rate
if not (phase_1_savepath).exists():
    model = model.to(device)
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
    print(f'Loaded model from {phase_1_savepath}')

# Save the trace pre-model pruning for size comparison
model = model.cpu().eval()
trace = torch.jit.trace(model, torch.ones(1, 3, 224, 224))
fname = Path(args.checkpoints_dir) / 'trace_pre_pruning.pth'
torch.jit.save(trace, str(fname))
print(f'Saved to {fname} - Size: ' + sizeof_fmt(fname.stat().st_size))

# Now we begin the main fine-tuning / pruning process
# We train 10 'pruning rounds' - in each round we train for some epochs,
# then at the end of the round we prune some of the weights from the model.
for i_pruning_round in range(args.num_pruning_rounds):
    model = model.to(device)

    # Finetune the entire model on our cats/dogs dataset. Use a bit of L1 regularisation
    # to encourage weights to go to 0, which should help with pruning.
    l1_reg = args.l1_reg_weight * (args.num_pruning_rounds - i_pruning_round) / args.num_pruning_rounds
    optimizer = SGDL1(model.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=l1_reg)
    criterion = nn.NLLLoss()
    scheduler = ReduceLROnPlateau(
        optimizer, factor=0.5, patience=2, cooldown=0, verbose=True)
    first_or_last_round = (
        i_pruning_round == 0 or i_pruning_round == (args.num_pruning_rounds - 1))

    # Train the model for some epochs
    val_acc_history = nn_utils.train_model(
        device, model, loader, criterion, optimizer,
        num_epochs=30 if first_or_last_round else args.num_epochs_per_pruning_round,
        postproc=lambda x: F.log_softmax(x, dim=1),
        scheduler=scheduler
    )
    model = model.cpu()
    savename_this = (
        Path(args.checkpoints_dir) /
        phase_2_savename.replace('.pth', ('_%02d.pth' % i_pruning_round))
    )
    torch.save(model.state_dict(), savename_this)
    print(f'Saved to {savename_this}')

    # Save the pruned + trained model
    model = model.cpu().eval()
    trace = torch.jit.trace(model, torch.ones(1, 3, 224, 224))
    fname = f'trace_pruned_{args.pruning_factor_per_round ** i_pruning_round:.3f}.pth'
    fname = Path(args.checkpoints_dir) / fname
    torch.jit.save(trace, str(fname))
    print(f'Saved to {fname} - Size: ' + sizeof_fmt(fname.stat().st_size))

    # Prune the smallest weights from the model
    if i_pruning_round < args.num_pruning_rounds - 1:
        model = prune_squeezenet(model, args.pruning_factor_per_round).to(device)
        print('Metrics after pruning:')
        nn_utils.eval_model(device, model, loader, criterion,
                            postproc=lambda x: F.log_softmax(x, dim=1))

# Save the final model to the root of the repo
copyfile(fname, Path('../final_pruned_model.pth'))
