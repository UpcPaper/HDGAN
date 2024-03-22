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
import sys

class ChestXrayDataSet(Dataset):
    def __init__(self,
                 image_dir,
                 caption_json,
                 file_list,
                 vocabulary,
                 s_max=6,
                 n_max=10,
                 transforms=None):
        self.image_dir = image_dir
        self.caption = JsonReader(caption_json)  # caption
        self.file_names, self.labels = self.__load_label_list(file_list)  # 图片名 和 label
        self.vocab = vocabulary
        self.transform = transforms
        self.s_max = s_max
        self.n_max = n_max

    def __load_label_list(self, file_list):  # 加载标签
        labels = []
        filename_list = []
        with open(file_list, 'r') as f:
            for line in f:
                items = line.split()
                image_name = items[0]  # 图片名
                label = items[1:]
                label = [int(i) for i in label]
                image_name = '{}.png'.format(image_name)
                # image_name = '{}'.format(image_name)
                filename_list.append(image_name)
                labels.append(label)
        return filename_list, labels

    def __getitem__(self, index):
        image_name = self.file_names[index]
        # Image.open()专接图片路径，用来直接读取该路径指向的图片。
        # 要求路径必须指明到哪张图，不能只是所有图所在的文件夹
        image = Image.open(os.path.join(self.image_dir, image_name)).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        try:
            text = self.caption[image_name]
        except Exception as err:
            text = 'normal. '
        # print(text)
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
        return image, image_name, list(label / np.sum(label)), target, sentence_num, max_word_num

    def __len__(self):
        return len(self.file_names)


def collate_fn(data):
    images, image_id, label, captions, sentence_num, max_word_num = zip(*data)
    images = torch.stack(images, 0)  # 叠加 第一维度

    max_sentence_num = max(sentence_num)  # 最大句子数
    max_word_num = max(max_word_num)  # 句子的最大单词数
    targets = np.zeros((len(captions), max_sentence_num, max_word_num + 1))
    prob = np.zeros((len(captions), max_sentence_num))

    for i, caption in enumerate(captions):
        for j, sentence in enumerate(caption):
            targets[i, j, :len(sentence)] = sentence[:]
            prob[i][j] = len(sentence) > 0
    return images, image_id, torch.Tensor(label), targets, prob


def get_loader(image_dir,
               caption_json,
               file_list,
               vocabulary,
               transform,
               batch_size,
               s_max=10,
               n_max=50,
               shuffle=True):
    dataset = ChestXrayDataSet(image_dir=image_dir,
                               caption_json=caption_json,
                               file_list=file_list,
                               vocabulary=vocabulary,
                               s_max=s_max,
                               n_max=n_max,
                               transforms=transform)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              collate_fn=collate_fn)
    return data_loader


if __name__ == '__main__':
    vocab_path = '../data/new_data/vocab.pkl'
    image_dir = '../data/images'
    caption_json = '../data/new_data/captions.json'
    file_list = '../data/new_data/train_data.txt'
    batch_size = 6
    resize = 256
    crop_size = 224

    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    data_loader = get_loader(image_dir=image_dir,
                             caption_json=caption_json,
                             file_list=file_list,
                             vocabulary=vocab,
                             transform=transform,
                             batch_size=batch_size,
                             shuffle=True)
    # c = ChestXrayDataSet(image_dir=image_dir,
    #                            caption_json=caption_json,
    #                            file_list=file_list,
    #                            vocabulary=vocab,
    #                            s_max=10,
    #                            n_max=30,
    #                            transforms=transform)
    # print("dd", c[5])

    for i, (image, image_id, label, target, prob) in enumerate(data_loader):
        print(i)
        print("image.shape", image.shape)
        print("image_id", image_id)
        print("label", label.shape)
        # print("target", target.shape)
        print("target.shape", target.shape)
        print("prob", prob)
        print("====================")
        # break
