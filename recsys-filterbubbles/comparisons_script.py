import torch
from torch.cuda import random
import torch.nn as nn
import torch.nn.functional as F
import os
import torch.optim as optim
from os import listdir
from os.path import isfile, join
from tracin.tracin_batched import save_tracin_checkpoint, load_tracin_checkpoint,  approximate_tracin_batched
import pandas as pd
from LSTM_clean.model import LSTM
import numpy as np
import re
from statistics import mean
import scipy.stats as stats
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from copy import deepcopy
from itertools import product
import time
import pickle
from comparison_helper import get_train_validation, get_train_subset, get_length, get_validation, get_points

OUTPUT_SIZE = 1743
NUM_TRAIN_SAMPLES = 50
NUM_VAL_SAMPLES = 50
NUM_REPETITIONS = 25
BATCH_SIZE = 4096
train_names = ["random", "diverse", "filter", "breaking"]
test_names = ["breaking", "filter"]

curr_dir = os.getcwd()
path = curr_dir + "/checkpoints_subset/"
checkpoints = []
with os.scandir(path) as listOfEntries:
    for entry in listOfEntries:
        # print all entries that are files
        if entry.is_file():
            checkpoints.append(os.path.join(path,entry.name))
            
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='4'

cpu_device = torch.device("cpu")
print("CPU Device is ", cpu_device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device is ", device)

combos = list(product(train_names, test_names))
print("Combos \n ", combos)
aggregate_influences = {}

for combo in combos:
    TRAIN_NAME = combo[0]
    TEST_NAME = combo[1]
    train, train_labels, valid, valid_labels = get_train_validation(train_dataset=TRAIN_NAME, valid_dataset=TEST_NAME)
    train_lengths = [get_length(i) for i in train]

    train_copy = deepcopy(train)
    train_labels_copy = deepcopy(train_labels)

    influences = []
    start_time = time.time()
    print("About to start running")
    for h in range(NUM_REPETITIONS):
        print(f"Starting outer loop with {h}")
        start_length_time = time.time()
        if len(train_copy) != 0:
            print("About to cartesian product")
            sources, source_labels, targets, target_labels = get_points(train_copy, train_labels_copy, valid, valid_labels, x_num_sample=NUM_TRAIN_SAMPLES, y_num_sample=NUM_VAL_SAMPLES, seed=h)
            print("About to tracin")
            influence = approximate_tracin_batched(LSTM, sources=sources, targets=targets, source_labels=source_labels, target_labels=target_labels, optimizer="SGD", paths=checkpoints, batch_size=BATCH_SIZE, num_items=OUTPUT_SIZE, device=device)
            influences.append(influence)
            end_length_time = time.time()
            print(f"Influence is : {influence} \nTime elapsed {end_length_time-start_length_time}")
        else:
            influences.append(-1)
        outer_end_time = time.time()

        print("_______________________________________________________________________________")


    influences = [float(i) for i in influences]
    print(f"Influences are \n{influences}")
    full_string = TRAIN_NAME + "_" + TEST_NAME
    aggregate_influences[full_string] = influences
    file_name = "train_"+ TRAIN_NAME + "_test_" + TEST_NAME +"_full.pkl" 

    with open(file_name, 'wb') as f:
        pickle.dump(influences, f)
    print(f"DONE WITH A COMBO {TRAIN_NAME} {TEST_NAME}")
    print("__________________________________________________________________________________")

for key1, val1 in aggregate_influences:
    for key2, val2 in aggregate_influences:
        print(f"Comparing {key1} and {key2}")
        print(f"Test statistics is \n{stats.ttest_ind(a=np.array(val1), b=np.array(val2), equal_var=False)}")
        print("____________________________________________")