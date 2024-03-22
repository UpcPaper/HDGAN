import pickle
import os
import sys
import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import numpy as np
from torchvision import transforms
import pickle
import matplotlib.pyplot as plt

if __name__ == '__main__':
    filename_list = []
    # with open("./train_data.txt", 'r') as f:
    #     for line in f:
    #         items = line.split()
    #         image_name = items[0]
    #         image_name = '{}.png'.format(image_name)
    #         filename_list.append(image_name)

    with open("/home/upc/sxx/Data/cov_ctr/annotation.json", 'r') as f:
        data = json.load(f)
    data_train = data['train']
    data_val = data['val']
    data_test = data['test']

    id = []
    labels = []
    for line in range(len(data_train)):
        image_name1 = data_train[line]['id']
        id.append(image_name1)
        labels.append(data_train[line]['report'])

    for line in range(len(data_val)):
        image_name1 = data_val[line]['id']
        id.append(image_name1)
        labels.append(data_val[line]['report'])

    for line in range(len(data_test)):
        image_name1 = data_test[line]['id']
        id.append(image_name1)
        labels.append(data_test[line]['report'])

    data = dict(zip(id, labels))

    # 将数据写入 JSON 文件
    with open("./all_true_data_cov.json", 'w') as json_file:
        json.dump(data, json_file)

