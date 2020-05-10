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

def get_dataset(data_dir):
    file_list = sorted(os.listdir(data_dir))
    dataset_file = []
    for file in file_list:
        if file.find("dataset") != -1:
            dataset_file.append(os.path.join(data_dir, file))
    return dataset_file
    
    
