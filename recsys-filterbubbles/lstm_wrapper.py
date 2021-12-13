import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from copy import deepcopy

from LSTM_clean.model import LSTM

## NOTE: For most experiments, just change SAVE_PREFIX, EPOCHS, and OUTPUT SIZE

# Data Location
SAVE_FOLDER = "/raid/home/myang349/recsys-filterbubbles/data/twitch_sequence/"
SAVE_PREFIX = ""
SAVE_TRAIN_NAME = SAVE_PREFIX + "train.data"
SAVE_VALID_NAME = SAVE_PREFIX + "valid.data"
SAVE_TEST_NAME = SAVE_PREFIX + "test.data"

# Configuration for MODEL
EPOCHS = 200

# OUTPUT_SIZE: Should be max itemid in re-indexed + 1 for the 0 item (i.e. # of unique items + 1)
# OUTPUT_SIZE = 5400
OUTPUT_SIZE = 3312

# Non-Changed
LEARNING_RATE = 5e-3
MOMENTUM = 0.9

def train_model():
    """Trains a model using the specified parameters at the top of the file"""

    # Setting Cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device is", device)

    # The format is:
    # N x 2 x (sequence, 
    train_data = np.load(os.path.join(SAVE_FOLDER, SAVE_TRAIN_NAME), allow_pickle=True)
    valid_data = np.load(os.path.join(SAVE_FOLDER, SAVE_VALID_NAME), allow_pickle=True)
    test_data = np.load(os.path.join(SAVE_FOLDER, SAVE_TEST_NAME), allow_pickle=True)

    print(f"Train: {len(train_data)}, Valid: {len(valid_data)}")

    # Output size should be # of unique items in data + 1 for the 0 item
    model = LSTM(input_size=128, output_size=OUTPUT_SIZE, hidden_dim=64, n_layers=1, device=device).to(device)
    model.LSTM.flatten_parameters()
    print("Model is ", model)

    print("\nTraining and testing")
    _, train_losses, test_losses, test_mrr, test_hits = model.traintest(train=train_data,test=valid_data, epochs=EPOCHS, learning_rate=LEARNING_RATE, momentum=MOMENTUM)
    print("\nFinished!")

    return model, train_losses, test_losses, test_mrr, test_hits

def train_model2(train_data, valid_data):
    """This version of train_model will take the data you want to use as parameters"""

    # Setting Cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device is", device)

    train_data = deepcopy(train_data)
    valid_data = deepcopy(valid_data)

    print(f"Train: {len(train_data)}, Valid: {len(valid_data)}")

    # Output size should be # of unique items in data + 1 for the 0 item
    model = LSTM(input_size=128, output_size=OUTPUT_SIZE, hidden_dim=64, n_layers=1, device=device).to(device)
    model.LSTM.flatten_parameters()
    print("Model is ", model)

    print("\nTraining and testing")
    _, train_losses, test_losses, test_mrr, test_hits = model.traintest(train=train_data,test=valid_data, epochs=EPOCHS, learning_rate=LEARNING_RATE, momentum=MOMENTUM)
    print("\nFinished!")

    return model, train_losses, test_losses, test_mrr, test_hits


def get_topk_predictions(model, data, k):
    data = deepcopy(data)

    # Generate embeddings and move to cuda
    embedded_data = []
    for pt in data:
        embedded_data.append(model.item_emb(torch.LongTensor(pt).to(model.device)))
    embedded_data = torch.stack(embedded_data, dim=0).detach()

    output, hidden = model.forward(embedded_data)
    preds = torch.topk(output, k).indices.tolist()
    return preds