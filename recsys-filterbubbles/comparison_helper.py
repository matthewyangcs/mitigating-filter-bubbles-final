import os
import numpy as np
from sklearn.utils import shuffle
from itertools import product

def get_checkpoints():
    curr_dir = os.getcwd()
    path = curr_dir + "/checkpoint_subsets/"
    checkpoints = []
    with os.scandir(path) as listOfEntries:
        for entry in listOfEntries:
            # print all entries that are files
            if entry.is_file():
                checkpoints.append(os.path.join(path,entry.name))
    return checkpoints


def get_train_validation(train_dataset, valid_dataset):
    cwd = os.getcwd()
    test_path = cwd + "/data/twitch_sequence/test.data"
    print(f'Train Dataset {train_dataset}')
    print(f"Valid dataset {valid_dataset}")
    if train_dataset == "filter":
        train_path = cwd + "/data/twitch_sequence/train_pts_filter_bubble.data"
    elif train_dataset == "diverse":
        train_path = cwd + "/data/twitch_sequence/train_pts_diverse.data"
    elif train_dataset == "random":
        train_path = cwd + "/data/twitch_sequence/train.data"
    else:
        train_path = cwd + "/data/twitch_sequence/train_pts_breaking_bubble.data"
    # Valid path
    if valid_dataset == "filter":
        valid_path = cwd + "/data/twitch_sequence/valid_pts_filter_bubble.data"
    elif valid_dataset == "random":
        valid_path = cwd + "/data/twitch_sequence/valid.data"
    else:
        valid_path = cwd + "/data/twitch_sequence/valid_pts_breaking_bubble.data"

    train_data = np.load(train_path, allow_pickle=True)
    valid_data = np.load(valid_path, allow_pickle=True)
    train = [t[0] for t in train_data]
    train_labels = [t[1] for t in train_data]
    valid = [t[0] for t in valid_data]
    valid_labels = [t[1] for t in valid_data]
    if train_dataset == "random":
        train, train_labels = shuffle(train, train_labels, random_state=69)
    print(f"Train set is length: {len(train)}")
    print(f"Validation set is length: {len(valid)}")
    return train, train_labels, valid, valid_labels

def get_train_subset(length, x, x_labels, train_lengths):
    x_subset =[]
    x_labels_subset = []
    for i in range(len(x)):
        if train_lengths[i] == length:
            x_subset.append(x[i])
            x_labels_subset.append(x_labels[i])
    # x_subset, _, x_labels_subset, _ = train_test_split(x_subset, x_labels_subset, train_size=num_sample, random_state=seed)
    # x_subset, x_labels_subset = shuffle(x_subset, x_labels_subset, random_state=seed)
    # if len(x_subset) > subset_size:
    #     return x_subset[:subset_size], x_labels_subset[:subset_size]
    # else:
    return x_subset, x_labels_subset

def get_length(data_point):
    for i in range(len(data_point)):
        if data_point[i] != 0:
            return 49 -i
    return 0


def get_validation(y, y_label, y_num_sample=10, seed=69):
    y, y_label = shuffle(y, y_label, random_state=seed)
    if len(y) > y_num_sample:
        # print("in y if")
        y, y_label = y[:y_num_sample], y_label[:y_num_sample]
    return y, y_label


def get_points(x, x_label, y, y_label,x_num_sample =100, y_num_sample=10, seed=69):
    # print(f"length of x is {len(x)}, {len(x_label)}")
    # print(f"length of y is {len(y)}, {len(y_label)}")
    x, x_label = shuffle(x, x_label, random_state= seed)
    y, y_label = shuffle(y, y_label, random_state=seed)
    if len(x) > x_num_sample:
        # print("In x if")
        x, x_label = x[:x_num_sample], x_label[:x_num_sample]
    if len(y) > y_num_sample:
        # print("in y if")
        y, y_label = y[:y_num_sample], y_label[:y_num_sample]
    combos = list(product(zip(x, x_label), zip(y, y_label)))
    # print(f"Length of combos is {len(combos)}")
    sources = [c[0][0] for c in combos]
    source_labels = [c[0][1] for c in combos]
    targets = [c[1][0] for c in combos]
    target_labels = [c[1][1] for c in combos]
    # print(f"Number of datapoints {len(sources)}")
    return sources, source_labels, targets, target_labels

