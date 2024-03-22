import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import os


class TxtDataset(Dataset):  # 这是一个Dataset子类
    def __init__(self):

        self.file_path = '../data/new_data/new_disc_train_fake_data.txt'
        self.count = 0
        input_data = open(self.file_path, 'r')
        caption = []
        label = []
        for line in input_data:
            self.count = self.count + 1
            line = line.strip('\n')
            line = line.rstrip()  # 删除 string 字符串末尾的指定字符（默认为空格）.
            # words = line.split(',')
            caption.append((line))
            # label.append(words[1])
        self.caption = caption
        # self.label = label
        input_data.close()
        self.max_len = self.get_max_num(self.caption)

    @staticmethod
    def get_max_num(caption):
        max_num = 0
        for k in range(self.count):
            spaces = 0
            spaces += caption[k].count(' ')
            count = spaces + 1
            if count > max_num:
                max_num = count
        return max_num

    @staticmethod
    def get_num(caption, index):
        num = caption[index].count(' ')
        return num+1

    def __getitem__(self, index):
        input_data = self.caption[index]
        # label = self.label[index]
        ori_length = self.get_num(self.caption, index)
        print(ori_length)
        s = " 0"
        if ori_length < self.max_len:
            for j in range(self.max_len - ori_length):
                input_data = input_data + s
        return input_data

    def __len__(self):
        return len(self.Data)

    @staticmethod
    def read_data(filepath):
        # file_path = os.path.join(file_path)
        data_file = open(filepath, 'r').readlines()
        test_loader = DataLoader(data_file, batch_size=1, shuffle=False,
                                 num_workers=0)
        for traindata in test_loader:
            print(traindata)


if __name__ == '__main__':
    # 创建一个TxtDataset对象,__getitem__的调用要通过： 对象[索引]调用
    data = TxtDataset()
    print(data.max_len)
    print("===================================")
    with open('../data/new_data/padded_disc_train_fake_data.txt', 'w') as f:
        for i in range(self.count):
            s = str(data[i]).replace('(', '').replace(')', '')  # 去除[],这两行按数据不同，可以选择
            s = s.replace("'", '') + '\n'  # 去除单引号,每行末尾追加换行符
            f.write(s)
        f.close()












