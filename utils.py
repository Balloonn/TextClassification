import pandas as pd
import numpy as np
from pickle_file_operator import PickleFileOperator
from params import *
from torch.utils.data import Dataset
import torch


def load_pk_file():
    labels = PickleFileOperator(file_path=LABELS_PK_PATH).read()
    chars = PickleFileOperator(file_path=CHARS_PK_PATH).read()
    label_dict = dict(zip(labels, range(len(labels))))
    chars_dict = dict(zip(chars, range(len(chars))))
    return label_dict, chars_dict


def load_csv_file(file_path):
    df = pd.read_csv(file_path)
    content, labels = [], []
    for index, row in df.iterrows():
        content.append(row['content'])
        labels.append(row['label'])
    return content, labels


def text_feature(content, labels, char_dict, label_dict):
    samples, y_real = [], []
    for s_content, s_label in zip(content, labels):
        samples.append(label_dict[s_content])
        train_sample = []
        for char in s_content:
            if char in char_dict:
                train_sample.append(START_NO + char_dict[char])
            else:
                train_sample.append(UNK_NO)
        if len(train_sample) < SEQ_LEN:
            samples.append(train_sample + ([PAD_NO] * (SEQ_LEN - len(train_sample))))
        else:
            samples.append(train_sample[:SEQ_LEN])
    return samples, y_real


class CSVDataset(Dataset):
    def __init__(self, file_path):
        label_dict, char_dict = load_pk_file()
        content, labels = load_csv_file(file_path)
        samples, y_real = text_feature(content, labels, char_dict, label_dict)
        self.x = torch.from_numpy(np.array(samples))
        self.y = torch.from_numpy(np.array(y_real))

