import os
import time
import copy
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

data_dir = 'dataset'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LinearRegression(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegression, self).__init__()
        self.linear1 = nn.Linear(input_dim, 100, bias=True)
        self.activate1 = nn.PReLU()
        self.linear2 = nn.Linear(100, 64, bias=True)
        self.activate2 = nn.PReLU()
        self.linear3 = nn.Linear(64, 32, bias=True)
        self.activate3 = nn.PReLU()
        self.linear4 = nn.Linear(32, 1, bias=True)
        self.model = nn.Sequential(self.linear1, self.activate1, self.linear2, self.activate2, self.linear3,
                                   self.activate3, self.linear4)

    def forward(self, x):
        out = self.model(x)
        return out


def train(model, criterion, optimizer, dataloader, num_epochs=200, verbose=True):
    history = {'train': [], 'val': []}
    since = time.time()
    best_loss = np.inf
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            num = 0
            # Iterate over data.
            for x, y in dataloader[phase]:
                num += x.size(0)
                x = x.to(device)
                y = y.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    estimation = model(x)
                    estimation = estimation.squeeze()
                    loss = criterion(estimation, y)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * x.size(0)

            epoch_loss = running_loss / num
            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))
            history[phase].append(epoch_loss)

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, history


def get_dataset(data_dir):
    file_list = sorted(os.listdir(data_dir))
    dataset_file = []
    for file in file_list:
        if file.find("dataset") != -1:
            dataset_file.append(os.path.join(data_dir, file))
    return dataset_file


errors = {'max_error': [], 'mean_error': [], 'max_relative_error': [], 'mean_relative_error': []}


def probability_distribution(data, name, bins=50):
    figure, (ax0, ax1) = plt.subplots(2, 1)
    ax0.hist(data, bins, facecolor='blue', edgecolor='black', alpha=0.75, weights=np.ones_like(data) / len(data))
    ax0.set_title(name + ' p_distributation')
    ax1.hist(data, bins, density=True, facecolor='yellowgreen', edgecolor='black', alpha=0.75, cumulative=True)
    ax1.set_title(name + ' sum_distribution')
    plt.show()
    figure.savefig(name + '.png', dpi=600, format='png')


def estimate_all():
    dataset_file = get_dataset(data_dir)
    x_datasets = []
    y_datasets = []
    for file in dataset_file:
        dataset = pd.read_csv(file, header=None, index_col=None)
        dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
        dataset.dropna(axis=0, inplace=True)
        x_dataset = dataset.iloc[:, :-1]
        y_dataset = dataset.iloc[:, -1]
        x_datasets.append(x_dataset)
        y_datasets.append(y_dataset)
    x_dataset = pd.concat(x_datasets, axis=0)
    y_dataset = pd.concat(y_datasets, axis=0)
    print('x dataset shape: {}, y shape: {}'.format(x_dataset.shape, y_dataset.shape))
    name = 'all_graph'
    input_dim = x_dataset.shape[1]
    model = LinearRegression(input_dim)
    model.to(device)
    x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, random_state=42, train_size=0.9)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, random_state=42, train_size=0.8)
    print('x train dataset shape: {}, y train shape: {}'.format(x_train.shape, y_train.shape))
    print('x val dataset shape: {}, y val shape: {}'.format(x_val.shape, y_val.shape))
    print('x test dataset shape: {}, y test shape: {}'.format(x_test.shape, y_test.shape))
    train_inputs = torch.from_numpy(x_train.values).type(torch.float)
    train_targets = torch.from_numpy(y_train.values).type(torch.float)
    val_inputs = torch.from_numpy(x_val.values).type(torch.float)
    val_targets = torch.from_numpy(y_val.values).type(torch.float)
    train_dataset = TensorDataset(train_inputs, train_targets)
    val_dataset = TensorDataset(val_inputs, val_targets)
    batch_size = 256
    dataloader = {'train': DataLoader(train_dataset, batch_size, shuffle=True),
                  'val': DataLoader(val_dataset, val_inputs.size(0), shuffle=True)}
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters())
    net, history = train(model, criterion, optimizer, dataloader, num_epochs=500)
    print('finish train on graph {}'.format(name))
    figure = plt.figure(name)
    plt.title("{}: loss".format(name))
    train_pl = plt.plot(history['train'], label='train_loss')
    val_pl = plt.plot(history['val'], label='val_loss')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.legend()
    # write_result(bt, G_info_list[i], end - start)
    figure.savefig("result_loss/{}.png".format(name), dpi=600, format='png')
    plt.show()
    torch.save({
        'model_weights': net.state_dict(),
        'history': history
    }, './models/{}_model.pkl'.format(name))
    estimation = net(torch.Tensor(x_test.values).type(torch.float).to(device))
    estimation = estimation.cpu().detach().numpy()
    estimation = np.round(np.squeeze(estimation))
    error_abs = np.abs(estimation - y_test.values)
    error_relative = np.abs(estimation - y_test.values) / y_test.values
    mask = np.isfinite(error_relative)
    error_relative = error_relative[mask]

    probability_distribution(error_relative, name + '_relative_error', 100)
    max_error = np.max(error_abs)
    mean_error = np.mean(error_abs)
    max_relative_error = np.max(error_relative)
    mean_relative_error = np.mean(error_relative)
    errors['max_error'].append(max_error)
    errors['mean_error'].append(mean_error)
    errors['max_relative_error'].append(max_relative_error)
    errors['mean_relative_error'].append(mean_relative_error)
    estimation = estimation.tolist()
    result_file = name + "_estimation.csv"
    with open(result_file, 'w+') as f:
        for idx, dist in enumerate(estimation):
            f.writelines(str(dist) + '\t' + str(y_test.values[idx]) + '\n')


def estimate_onebyone():
    dataset_file = get_dataset(data_dir)
    historys = {}
    models = {}
    for file in dataset_file:
        name = file.split('_')[0].split('/')[1]
        dataset = pd.read_csv(file, header=None, index_col=None)
        dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
        dataset.dropna(axis=0, inplace=True)
        x_dataset = dataset.iloc[:, :-1]
        y_dataset = dataset.iloc[:, -1]
        print('x dataset shape: {}, y shape: {}'.format(x_dataset.shape, y_dataset.shape))
        input_dim = x_dataset.shape[1]
        model = LinearRegression(input_dim)
        model.to(device)
        x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, random_state=42, train_size=0.9)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, random_state=42, train_size=0.8)
        print('x train dataset shape: {}, y train shape: {}'.format(x_train.shape, y_train.shape))
        print('x val dataset shape: {}, y val shape: {}'.format(x_val.shape, y_val.shape))
        print('x test dataset shape: {}, y test shape: {}'.format(x_test.shape, y_test.shape))
        train_inputs = torch.from_numpy(x_train.values).type(torch.float)
        train_targets = torch.from_numpy(y_train.values).type(torch.float)
        val_inputs = torch.from_numpy(x_val.values).type(torch.float)
        val_targets = torch.from_numpy(y_val.values).type(torch.float)
        train_dataset = TensorDataset(train_inputs, train_targets)
        val_dataset = TensorDataset(val_inputs, val_targets)
        batch_size = 128
        dataloader = {'train': DataLoader(train_dataset, batch_size, shuffle=True),
                      'val': DataLoader(val_dataset, val_inputs.size(0), shuffle=True)}
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters())
        net, history = train(model, criterion, optimizer, dataloader, num_epochs=200)
        print('finish train on graph {}'.format(name))
        figure = plt.figure(name)
        plt.title("{}: loss".format(name))
        train_pl = plt.plot(history['train'], label='train_loss')
        val_pl = plt.plot(history['val'], label='val_loss')
        plt.xlabel('Epochs')
        plt.ylabel('loss')
        plt.legend()
        # write_result(bt, G_info_list[i], end - start)
        figure.savefig("result_loss/{}.png".format(name), dpi=600, format='png')
        plt.show()
        torch.save({
            'model_weights': net.state_dict(),
            'history': history
        }, './models/{}_model.pkl'.format(name))
        estimation = net(torch.Tensor(x_test.values).type(torch.float).to(device))
        estimation = estimation.cpu().detach().numpy()
        estimation = np.round(np.squeeze(estimation))
        error_abs = np.abs(estimation - y_test.values)
        error_relative = np.abs(estimation - y_test.values) / y_test.values
        mask = np.isfinite(error_relative)
        error_relative = error_relative[mask]

        probability_distribution(error_relative, name + '_relative_error', 100)
        max_error = np.max(error_abs)
        mean_error = np.mean(error_abs)
        max_relative_error = np.max(error_relative)
        mean_relative_error = np.mean(error_relative)
        errors['max_error'].append(max_error)
        errors['mean_error'].append(mean_error)
        errors['max_relative_error'].append(max_relative_error)
        errors['mean_relative_error'].append(mean_relative_error)
        estimation = estimation.tolist()
        result_file = name + "_estimation.csv"
        with open(result_file, 'w+') as f:
            for idx, dist in enumerate(estimation):
                f.writelines(str(dist) + '\t' + str(y_test.values[idx]) + '\n')


if __name__ == '__main__':
    estimate_onebyone()
    estimate_all()
