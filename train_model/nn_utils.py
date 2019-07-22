from fastprogress import progress_bar, master_bar
import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def pretrained_squeezenet_with_two_output_classes():
    model = models.squeezenet1_0(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Conv2d(512, 2, 1),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        DropSpatialDims(),
    )
    return model


class DropSpatialDims(nn.Module):
    def forward(self, input):
        return input[:, :, 0, 0]


def train_model(device, model, dataloaders, criterion, optimizer,
                num_epochs=10, postproc=None, phases=['train', 'val'], scheduler=None):
    val_acc_history = []

    mb = master_bar(range(num_epochs))
    for epoch in mb:
        if num_epochs > 1:
            print('\nEpoch {}/{}'.format(epoch + 1, num_epochs))
            print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in phases:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in progress_bar(dataloaders[phase], parent=mb):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    if postproc is not None:
                        outputs = postproc(outputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'val':
                val_acc_history.append([epoch_acc, epoch_loss])
                if scheduler is not None and 'train' in phases:
                    try:
                        scheduler.step()
                    except Exception as _:
                        scheduler.step(epoch_loss)

    return val_acc_history


def eval_model(device, model, loader, criterion, postproc=None, phases=['val'], optimizer=None):
    if optimizer is None:
        optimizer = torch.optim.SGD([], 1e-8)
    return train_model(device, model, loader, criterion, optimizer,
                       num_epochs=1, postproc=postproc, phases=phases)
