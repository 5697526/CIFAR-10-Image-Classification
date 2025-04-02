import numpy as np
import os
import pickle


def load_data(data_dir='./data'):
    train_data, train_labels = [], []
    for i in range(1, 6):
        with open(os.path.join(data_dir, f'cifar-10-batches-py/data_batch_{i}'), 'rb') as f:
            batch = pickle.load(f, encoding='latin1')
            train_data.append(batch['data'])
            train_labels.extend(batch['labels'])

    with open(os.path.join(data_dir, 'cifar-10-batches-py/test_batch'), 'rb') as f:
        batch = pickle.load(f, encoding='latin1')
        test_data = batch['data']
        test_labels = batch['labels']

    train_data = np.vstack(train_data).astype(np.float32) / 255.0
    test_data = np.array(test_data, dtype=np.float32) / 255.0

    return train_data, np.array(train_labels), test_data, np.array(test_labels)


def preprocess_data(train_data, test_data):
    mean = np.mean(train_data, axis=0)
    std = np.std(train_data, axis=0)
    train_data = (train_data-mean) / (std+1e-7)
    test_data = (test_data-mean) / (std+1e-7)

    return train_data, test_data


def validation(train_data, train_labels, val_ratio=0.1):
    num_val = int(val_ratio*train_data.shape[0])
    indices = np.random.permutation(train_data.shape[0])
    val_idx, train_idx = indices[:num_val], indices[num_val:]

    return train_data[train_idx], train_labels[train_idx], train_data[val_idx], train_labels[val_idx]
