import codecs
from os import replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.model_selection import train_test_split

def create_validation_dataset(test: np.ndarray, val_size: float, random_state: int) -> Tuple[np.ndarray, np.ndarray]:
    users = test[:, 0]
    items = test[:, 1]
    ratings = test[:, 2]
    val = []
    test = []

    for user in set(users):
        indices = users == user
        pos_items = items[indices]        
        val_items = np.random.RandomState(random_state).choice(pos_items, int(val_size*len(pos_items)), replace=False)
        test_items = np.setdiff1d(pos_items, val_items)
        for val_item in val_items:
            item_indices = (items == val_item) & (users == user)
            val_rating = int(ratings[item_indices])
            val.append([user, val_item, val_rating])
        for test_item in test_items:
            item_indices = (items == test_item) & (users == user)
            test_rating = int(ratings[item_indices])
            test.append([user, test_item, test_rating])

    val = np.array(val)
    test = np.array(test)

    return test, val

def preprocess_dataset(data: str, threshold: int = 4, alpha: float = 0.5) -> Tuple:
    """Load and Preprocess datasets."""
    # load dataset.
    if data == 'yahoo':
        col = {0: 'user', 1: 'item', 2: 'rate'}
        with codecs.open(f'./data/yahoo/train.txt', 'r', 'utf-8', errors='ignore') as f:
            data_train = pd.read_csv(f, delimiter='\t', header=None)
            data_train.rename(columns=col, inplace=True)
        with codecs.open(f'./data/yahoo/test.txt', 'r', 'utf-8', errors='ignore') as f:
            data_test = pd.read_csv(f, delimiter='\t', header=None)
            data_test.rename(columns=col, inplace=True)

        data_train.user, data_train.item = data_train.user - 1, data_train.item - 1
        data_test.user, data_test.item = data_test.user - 1, data_test.item - 1

    elif data == 'coat':
        cols = {'level_0': 'user', 'level_1': 'item', 2: 'rate', 0: 'rate'}
        with codecs.open(f'./data/coat/train.ascii', 'r', 'utf-8', errors='ignore') as f:
            data_train = pd.read_csv(f, delimiter=' ', header=None)
            data_train = data_train.stack().reset_index().rename(columns=cols)
            data_train = data_train[data_train.rate != 0].reset_index(drop=True)
        with codecs.open(f'./data/coat/test.ascii', 'r', 'utf-8', errors='ignore') as f:
            data_test = pd.read_csv(f, delimiter=' ', header=None)
            data_test = data_test.stack().reset_index().rename(columns=cols)
            data_test = data_test[data_test.rate != 0].reset_index(drop=True)

    num_users, num_items = max(data_train.user.max()+1, data_test.user.max()+1), max(data_train.item.max()+1, data_test.item.max()+1)

    # binalize rating.
    data_train.rate[data_train.rate < threshold] = 0
    data_train.rate[data_train.rate >= threshold] = 1
    data_test.rate[data_test.rate < threshold] = 0
    data_test.rate[data_test.rate >= threshold] = 1
        
    print(data_train)
    print(data_test)

    # train-val-test, split
    train, test = data_train.values, data_test.values
    train, val = create_validation_dataset(train, val_size=0.3, random_state=12345)

    # train data freq
    item_freq = np.zeros(num_items, dtype=int)
    for tmp in train:
        if tmp[2] == 1:
            item_freq[int(tmp[1])] += 1

    # for training, only tr's ratings frequency used
    pscore = (item_freq / item_freq.max()) ** alpha

    # validation data freq
    # for testing
    for tmp in val:
        if tmp[2] == 1:
            item_freq[int(tmp[1])] += 1

    item_freq = item_freq**1.5 # pop^{(1+2)/2} gamma = 2

    # only positive data
    train = train[train[:, 2] == 1, :2]

    # creating training data
    all_data = pd.DataFrame(
        np.zeros((num_users, num_items))).stack().reset_index()
    all_data = all_data.values[:, :2]
    unlabeled_data = np.array(
        list(set(map(tuple, all_data)) - set(map(tuple, train))), dtype=int)
    train = np.r_[np.c_[train, np.ones(train.shape[0])],
                np.c_[unlabeled_data, np.zeros(unlabeled_data.shape[0])]]

    # save datasets
    path_data = Path(f'./data/{data}')
    point_path = path_data / f'point_{alpha}'
    point_path.mkdir(parents=True, exist_ok=True)

    # pointwise
    np.save(file=point_path / 'train.npy', arr=train.astype(np.int))
    np.save(file=point_path / 'val.npy', arr=val.astype(np.int))
    np.save(file=point_path / 'test.npy', arr=test.astype(np.int))
    np.save(file=point_path / 'pscore.npy', arr=pscore)
    np.save(file=point_path / 'item_freq.npy', arr=item_freq)
