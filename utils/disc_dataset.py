# -*- coding:utf-8 -*-

import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import json
from utils.build_vocab import Vocabulary, JsonReader
import numpy as np
from torchvision import transforms
import pickle


class DiscDataSet(Dataset):
    def __init__(self,
                 text_path,
                 vocabulary,
                 s_max=6,
                 n_max=10):
        self.vocab = vocabulary
        self.text_path = text_path
        self.s_max = s_max
        self.n_max = n_max
        # self.text_list = self.list_of_text(self.text_path)

    def list_of_text(self, path):
        list = []
        with open(path, 'r') as text:
            line = text.readline()
            while line:
                list.append(str(line))
        return list

    def __getitem__(self, index):
        lines = []
        with open(self.text_path, 'r') as file:
            while True:
                line = file.readline()
                if not line:
                    break
                line = line.strip('\n')
                lines.append(line)
        text = lines[index]

        target = list()
        max_word_num = 0

        for i, sentence in enumerate(text.split('. ')):  # 以什么分割 “.空格”
            if i >= self.s_max:
                break  # 跳出整个循环
            sentence = sentence.split()  # 再把每个句子分开
            if len(sentence) == 0 or len(sentence) == 1 or len(sentence) > self.n_max:
                continue  # 语句跳出本次循环，继续下一轮的循环，过滤掉0个，1个，超过n_max个单词的句子
            tokens = list()
            tokens.append(self.vocab('<start>'))
            tokens.extend([self.vocab(token) for token in sentence])
            tokens.append(self.vocab('<end>'))
            if max_word_num < len(tokens):
                max_word_num = len(tokens)
            target.append(tokens)
        sentence_num = len(target)  # 句子总数

        return target, sentence_num, max_word_num  # 一个caption的总单词数

    def __len__(self):
        text = open(self.text_path, 'r')
        counter =len(text.readlines())
        return counter


def collate_fn(data):
    captions, sentence_num, max_word_num= zip(*data)
    max_sentence_num = max(sentence_num)  # 最大句子数

    max_word_num = max(max_word_num)  # 最大单词数
    # targets = np.zeros((len(inputs), 6, 20))
    targets = np.zeros((len(captions), max_sentence_num, max_word_num + 1))
    for i, caption in enumerate(captions):
        for j, sentence in enumerate(caption):
            targets[i, j, :len(sentence)] = sentence[:]
    return targets


def get_loader(text_path, vocabulary, batch_size, s_max=6, n_max=30, shuffle=True):
    dataset = DiscDataSet(text_path, vocabulary, s_max=s_max, n_max=n_max,)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              drop_last=True,
                                              collate_fn=collate_fn)
    return data_loader


if __name__ == '__main__':
    vocab_path = '../data/new_data/vocab.pkl'
    path = '../data/new_data/disc_train_fake_data.txt'
    batch_size = 20

    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    data_loader = get_loader(text_path=path,
                             vocabulary=vocab,
                             batch_size=batch_size,
                             shuffle=True)
    for i, inputs in enumerate(data_loader):
        print("i=", i)
        print("inputs", inputs)


