import os, argparse, yaml, time, copy
import pandas as pd
import albumentations

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import models


from dataset.plant_doc import ImageFolder


def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, device, num_epochs=25 ):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_acc, val_acc = [], []
    train_loss, val_loss = [], []


    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            if phase=='train':
                train_acc.append(epoch_acc.cpu().detach().numpy())
                train_loss.append(epoch_loss)
            else:
                val_acc.append(epoch_acc.cpu().detach().numpy())
                val_loss.append(epoch_loss)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    pd.DataFrame({'train_acc':train_acc, 'val_acc': val_acc,
                'train_loss': train_loss, 'val_loss': val_loss}).to_csv('logs/run_log.csv', index=False)
    

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def main(config):
    print(config)
    train_data = ImageFolder(config['dataset_path'], split = 'Train', augment = True)
    print(f'{len(train_data)} labeled training images found!' )

    test_data = ImageFolder(config['dataset_path'], split = 'Test', augment = False)
    print(f'{len(test_data)} labeled testing images found!' )
    dataset_sizes = {'train': len(train_data), 'val': len(test_data)}


    train_dl = torch.utils.data.DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    test_dl = torch.utils.data.DataLoader(test_data, batch_size=config['batch_size'], shuffle=True, num_workers=4)

    dataloaders = {'train': train_dl, 'val': test_dl}

    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')


    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, config['n_class'])

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    if config['optimizer'] == 'adam':
        optimizer_ft = optim.Adam(model_ft.parameters(), lr=config['lr'])
    else:
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=config['lr'], momentum=config['momentum'])
   
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    

    model_ft = train_model(model_ft, dataloaders, dataset_sizes, criterion, optimizer_ft, exp_lr_scheduler,
                       device,  num_epochs=config['epochs'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    main(config)