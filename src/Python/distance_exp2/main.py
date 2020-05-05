import copy
import os
import time
import operator
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

data_dir = 'dataset'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 8 layers
class LinearRegression(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegression, self).__init__()
        self.activation = nn.ReLU()
        self.linear1 = nn.Sequential(nn.Linear(input_dim, 100), self.activation)
        self.linear2 = nn.Sequential(nn.Linear(100, 100), self.activation)
        self.linear3 = nn.Sequential(nn.Linear(100, 64), self.activation)
        self.linear4 = nn.Sequential(nn.Linear(64, 64), self.activation)
        self.linear5 = nn.Sequential(nn.Linear(64, 64), self.activation)
        self.linear6 = nn.Sequential(nn.Linear(64, 32), self.activation)
        self.linear7 = nn.Sequential(nn.Linear(32, 16), self.activation)
        self.linear8 = nn.Sequential(nn.Linear(16, 1), self.activation)
        self.model = nn.Sequential(self.linear1, self.linear2, self.linear3, self.linear4, self.linear5, self.linear6,
                                   self.linear7, self.linear8)

    def forward(self, x):
        out = self.model(x)
        return out


class ReDistanceTool():
    def __init__(self, graph_path, graph_name, graph_type):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("device:{}".format(self.device))
        self.train_split = 0.8
        
        self.graph_path = graph_path
        self.graph_type = graph_type  # 0: unweighted 1: weighted
        self.graph_name = graph_name
        # use id to save
        self.label_to_id_map = dict()
        self.beacons_start = 0
        # how to concat
        self.src_set = list()  # const train pairs
        self.des_set = list()
        self.init_beacons_num = 50
        self.max_beacons_num = 200
        self.iter_beacons_num = 10
        self.max_relative_error = 1e-2
        self.train_size = 1000
        self.beacons = set()
        self.graph = None
        self.dis_data = None
        self.dis_value = None
        self.errors = {'max_error': [], 'mean_error': [], 'max_relative_error': [], 'mean_relative_error': []}
        np.random.seed(42)
        self.init_graph()
        self.label_to_id_map = {list(self.graph.nodes())[index]:index for index in range(self.graph.order())}
        self.init_beacons()
        self.init_train_pairs()
        time1 = time.time()
        self.distance_dict = dict(nx.all_pairs_shortest_path_length(self.graph))  # iterator with dictionary keyed by target and shortest path length as the key value
        time2 = time.time()
        print("compute cost {:2f} s".format(time2 - time1))
        time1 = time.time()
        self.distance_dict_numpy = self.prepare_distance_numpy()
        time2 = time.time()
        print("construct cost {:2f} s".format(time2 - time1))
        print(self.distance_dict_numpy)

    def init_graph(self):
        if self.graph_type == 0:
            self.graph = nx.read_edgelist(self.graph_path, comments="#", delimiter=None, create_using=nx.Graph, nodetype= int)
        elif self.graph_type == 1:
            self.graph = nx.read_weighted_edgelist(self.graph_path, comments="#", delimiter=None, create_using=nx.Graph, nodetype= int)

    def init_beacons(self):
        self.beacons = None
        beacons_id = np.random.randint(0, self.graph.order(), self.init_beacons_num).tolist()
        self.beacons = set(beacons_id)

    def init_train_pairs(self):
        self.src_set.clear()
        self.des_set.clear()
        id_src = []
        id_des = []
        if self.train_size <= self.graph.order():
            id_src = np.random.permutation(self.graph.order())[:self.train_size]
            id_des = np.random.permutation(self.graph.order())[:self.train_size]
        for id in range(self.train_size):
            src_id = id_src[id]
            des_id = id_des[id]
            if src_id != des_id:
                self.src_set.append(src_id)
                self.des_set.append(des_id)
        print("get %d valid pairs" % len(self.src_set))

    def prepare_distance_numpy(self):
        # for node_src in range(self.graph.order()):
        #     for node_des in range(self.graph.order()):
        #         self.distance_dict_numpy[node_src, node_des] = self.distance_dict[list(self.graph.nodes())[node_src]][list(self.graph.nodes())[node_des]]
        ## The loop costs much time, refer to https://stackoverflow.com/questions/54021117/convert-dictionary-to-numpy-array we can change it
        # if the dictionary is inserted with the same order of graph.nodes() which means that array[0,1] represent the distance from g.nodes[0] to g.nodes[1]
        # ans = np.array([list(item.values()) for item in self.distance_dict.values()])
        # if the above dictionary are not satisfied, we can use the operator itemgetter
        getter = operator.itemgetter(*list(self.graph.nodes()))
        ans = np.array([getter(item) for item in getter(self.distance_dict)])
        return ans

    def prepare_train_set(self):
        dis_src = self.distance_dict_numpy[np.ix_(self.src_set, list(self.beacons))]
        dis_des = self.distance_dict_numpy[np.ix_(self.des_set, list(self.beacons))]
        dis_data = np.concatenate((dis_src, dis_des), axis=1)
        self.dis_data = dis_data
        self.dis_value = self.distance_dict_numpy[self.src_set, self.des_set]

    def obtain_beacons(self, worst_paris, beacons_num, sampler = 0):
        scores = dict.fromkeys(self.graph, 0.0) # beacons score
        samplers = [0, 1, 2]
        if sampler not in samplers:
            raise Exception("Not supported sampler")
        hyedges = []
        nodes = list(self.graph.nodes())
        for (src_id, des_id) in worst_paris:
            src = nodes[src_id]
            des = nodes[des_id]
            try:
                paths = nx.all_shortest_paths(self.graph, source=src, target=des)
                paths = [list(path) for path in paths]
            except nx.NetworkXNoPath:
                continue
            if sampler == 0:  ## hyedge
                hyedge = list(paths[random.randint(0, len(paths) - 1)])
                hyedge.remove(src)
                hyedge.remove(des)
                for node in hyedge:
                    scores[node] += 1
            elif sampler == 1: ## original hyedge
                hyedge = list(paths[random.randint(0, len(paths) - 1)])
                hyedge.remove(src)
                hyedge.remove(des)
                if len(hyedge) > 0:
                    hyedges.append(hyedge)
            elif sampler == 2:  ## yalg
                hyedge = []
                num_path = 0
                for path in paths:
                    num_path += 1
                    path.remove(src)
                    path.remove(des)
                    hyedge += path
                for node in hyedge:
                    scores[node] += 1 / num_path

        if sampler == 1:
            while len(hyedges) > 1:
                count = {}
                max_degree = 0
                max_node = 0
                for hyedge in hyedges:
                    for node in hyedge:
                        value = count.get(node, 0)
                        count[node] = value + 1
                for node, degree in count.items():
                    if degree >= max_degree:
                        max_degree = degree
                        max_node = node
                scores[max_node] = len(hyedges)
                # remove
                hyedges = [hyedge for hyedge in hyedges if max_node not in hyedge]
                # for hyedge in hyedges:
                #     for node in hyedge:
                #         bc_estimation[node]  += step
        score_array = np.array([scores[nodes[id]] for id in range(self.graph.order())])
        order1 = np.argsort(-score_array)
        return order1.tolist()[:beacons_num]
    
    def add_beacons(self, target_pairs):
        ## add the most often occured nodes
        new_beacons_id = set(self.obtain_beacons(target_pairs, beacons_num = self.iter_beacons_num, sampler = 0))
        orign_num = len(self.beacons)
        self.beacons  |=new_beacons_id
        now_num = len(self.beacons)
        print("add {} new beacons".format(now_num - orign_num))
    
    def probability_distribution(data, name, bins=50):
        figure, (ax0, ax1) = plt.subplots(2, 1)
        ax0.hist(data, bins, facecolor='blue', edgecolor='black', alpha=0.75, weights=np.ones_like(data) / len(data))
        ax0.set_title(name + ' p_distributation')
        ax1.hist(data, bins, density=True, facecolor='yellowgreen', edgecolor='black', alpha=0.75, cumulative=True)
        ax1.set_title(name + ' sum_distribution')
        plt.show()
        figure.savefig(name + '.png', dpi=600, format='png')
        
        
    def draw(self, history, estimation, y_test):
        errors = {'max_error': [], 'mean_error': [], 'max_relative_error': [], 'mean_relative_error': []}
        name = self.graph_name+ "_"+ str(len(self.beacons))
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
#         torch.save({
#             'model_weights': net.state_dict(),
#             'history': history
#         }, './models/{}_model.pkl'.format(name))
        estimation = np.round(np.squeeze(estimation))
        error_abs = np.abs(estimation - y_test)
        error_relative = np.abs(estimation - y_test) / y_test
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
                f.writelines(str(dist) + '\t' + str(y_test[idx]) + '\n')

    def train(self, num_epochs=200, batch_size = 128, verbose=True):
        epoch_max_loss = 1e-2
        epoch_max_diff_loss = 1e-4

        criterion = nn.MSELoss()

        while(len(self.beacons) <= self.max_beacons_num):
            # prepare data loader
            model =  LinearRegression(len(self.beacons)*2)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            model.to(self.device)
            self.prepare_train_set()
            print('x dataset shape: {}, y shape: {}'.format(self.dis_data.shape, self.dis_value.shape))
            train_num = (int)(len(self.dis_data)*self.train_split)
            train_data = self.dis_data[:train_num]
            train_value = self.dis_value[:train_num]
            val_data = self.dis_data[train_num:]
            val_value = self.dis_value[train_num:]
            train_inputs = torch.from_numpy(train_data).type(torch.float)
            train_targets = torch.from_numpy(train_value).type(torch.float)
            val_inputs = torch.from_numpy(val_data).type(torch.float)
            val_targets = torch.from_numpy(val_value).type(torch.float)
            train_dataset = TensorDataset(train_inputs, train_targets)
            dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
            history = {'train': [], 'val': []}
            since = time.time()
            best_train_loss = np.inf
            best_val_loss = np.inf
            # best_model_wts = copy.deepcopy(model.state_dict())
            epoch_losses = []
            for epoch in range(num_epochs):
                print('Epoch {}/{}'.format(epoch + 1, num_epochs))
                print('-' * 10)

                # Each epoch has a training and validation phase
                model.train()
                running_loss = 0.0
                num = 0
                # Iterate over data.
                for x, y in dataloader:
                    num += x.size(0)
                    x = x.to(self.device)
                    y = y.to(self.device)
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(True):
                        estimation = model(x)
                        estimation = estimation.squeeze()
                        loss = criterion(estimation, y)
                        # backward + optimize only if in training phase
                        loss.backward()
                        optimizer.step()
                    # statistics
                    running_loss += loss.item() * x.size(0)
                ## val
                model.eval()
                val_inputs = val_inputs.to(self.device)
                pred_v = model(val_inputs)
                pred_v = pred_v.squeeze()
                pred_s_v = pred_v.cpu().detach().numpy()
                val_targets_s = val_targets.numpy()
                final_val_loss = np.mean(np.power(pred_s_v - val_targets_s, 2))
                history["val"].append(final_val_loss)
                epoch_loss = running_loss / num
                history["train"].append(epoch_loss)
                print('train Loss: {:.4f}'.format(epoch_loss))
                if epoch_loss <= best_train_loss:
                    best_train_loss = epoch_loss
                epoch_losses.append(epoch_loss)
                # early stopping
                # if len(epoch_losses) >= 4:
                #     np_losses = np.array(epoch_losses[-4:0])
                #     if np.max(np_losses) < epoch_max_loss and np.max(np.diff(np_losses)) < epoch_max_diff_loss:
                #         print("early stopping at epoch {}".format(epoch))
                #         break

                # deep copy the model

                print()

            model.eval()
            train_inputs = train_inputs.to(self.device)
            pred = model(train_inputs)
            pred = pred.squeeze()
            pred_s = pred.cpu().detach().numpy()
            train_targets_s = train_targets.numpy()
            final_train_loss = np.mean(np.power(pred_s - train_targets_s, 2))
            max_pair_s = np.argsort(-(np.power(pred_s - train_targets_s, 2)))[-20:].tolist()

            node_color = ['r'] * self.graph.order()
            for pair in max_pair_s:
                src = self.src_set[pair]
                des = self.des_set[pair]
                node_color[src] = 'b'
                node_color[des] = 'b'
            nx.draw_networkx_nodes(self.graph, node_size=5, node_shape="o",edge_size=1, with_labels=True, pos=nx.spring_layout(self.graph), node_color= node_color)
            plt.show()
            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
            val_inputs = val_inputs.to(self.device)
            pred_v = model(val_inputs)
            pred_v = pred_v.squeeze()
            pred_s_v = pred_v.cpu().detach().numpy()
            val_targets_s = val_targets.numpy()
            final_val_loss = np.mean(np.power(pred_s_v - val_targets_s, 2))
            self.draw(history, pred_s_v, val_targets_s)
            print('Best train loss: {:4f} for # {} beacons'.format(best_train_loss, len(self.beacons)))
            print('Final train loss: {:4f} for # {} beacons'.format(final_train_loss, len(self.beacons)))
            print('Final val loss: {:4f} for # {} beacons'.format(final_val_loss, len(self.beacons)))
            start = time.time()
            new_src = [self.src_set[index] for index in max_pair_s]
            new_des = [self.des_set[index] for index in max_pair_s]
            self.add_beacons(zip(new_src, new_des))
            print("add beacons cost %.2f s" % (time.time() - start))
            # load best model weights
            # model.load_state_dict(best_model_wts)
            # return model, history

    def get_dataset(data_dir):
        file_list = sorted(os.listdir(data_dir))
        dataset_file = []
        for file in file_list:
            if file.find("dataset") != -1:
                dataset_file.append(os.path.join(data_dir, file))
        return dataset_file

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
    graph_path = "dataset/graph/facebook_combined.txt"
    print("begin to train")
    start = time.time()
    tool = ReDistanceTool(graph_path = graph_path, graph_name = "facebook_combined", graph_type = 0)
    print("prepare cost %.2f s" % (time.time() - start))
    tool.train()