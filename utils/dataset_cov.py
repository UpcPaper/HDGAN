import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import numpy as np
from torchvision import transforms
import pickle
import matplotlib.pyplot as plt
import os

class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.id2word = {}
        self.idx = 0
        self.add_word('<pad>')  # 0
        self.add_word('<start>')  # 1
        self.add_word('<end>')  # 2
        self.add_word('<unk>')  # 3

    def add_word(self, word):
        if word not in self.word2idx:

            self.word2idx[word] = self.idx
            self.id2word[self.idx] = word
            self.idx += 1

    def get_word_by_id(self, id):
        # print(self.id2word[id])
        return self.id2word[id]

    def __call__(self, word):
        if word not in self.word2idx:
            # print(word)
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        # print(self.word2idx)
        # print(self.id2word)
        return len(self.word2idx)

class ChestXrayDataSet(Dataset):
    def __init__(self,
                 data_dir,
                 split,
                 vocabulary,
                 transforms,
                 s_max=7,
                 n_max=30):

        self.vocab = vocabulary
        self.ann_path = '/home/upc/sxx/Data/cov_ctr/annotation.json'
        self.transform = transforms
        self.s_max = s_max
        self.n_max = n_max
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.file_names, self.caption = self.__load_label_list(data_dir, split)  # 图片名 和 label

        self.examples = self.ann[split]
        # for i in range(len(self.examples)):
        #     self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
        #     self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])

    def __len__(self):
        return len(self.examples)

    def __load_label_list(self, file_list, split):  # 加载标签
        labels = []
        image1 = []
        with open(file_list, 'r') as f:
            data = json.load(f)
        data_all = data[split]

        for line in range(len(data_all)):
            image_name1 = data_all[line]['image_path'][0]
            image1.append(image_name1)
            labels.append(data_all[line]['report'])
        return image1, labels

    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        DATA_PATH = "/home/upc/sxx/Data/cov_ctr/images/"
        image1 = Image.open(os.path.join(DATA_PATH, image_path)).convert('RGB')
        image2 = Image.open(os.path.join(DATA_PATH, image_path)).convert('RGB')
        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
            text = self.caption[idx]
            text = text.lower()
            text = text.replace(',', '')
        target = list()
        max_word_num = 0
        for i, sentence in enumerate(text.split('. ')):  # 以什么分割 “.空格”
            if i >= self.s_max:
                break  # 跳出整个循环
            sentence = sentence.split()  # 再把每个句子分开
            if len(sentence) == 0 or len(sentence) == 1 or len(sentence) > self.n_max-2:
                continue  # 语句跳出本次循环，继续下一轮的循环，过滤掉0个，1个，超过n_max个单词的句子
            tokens = list()
            tokens.append(self.vocab('<start>'))
            tokens.extend([self.vocab(token) for token in sentence])
            tokens.append(
                self.vocab('<end>'))
            if max_word_num < len(tokens):
                max_word_num = len(tokens)
            target.append(tokens)
        sentence_num = len(target)  # 句子总数
        #
        # report_ids = example['ids']
        # report_masks = example['mask']
        # seq_length = len(report_ids)
        return image1, image2, target, sentence_num, max_word_num, image_id

def collate_fn(data):
    image1, image2, captions, sentence_num, max_word_num, id = zip(*data)
    images1 = torch.stack(image1, 0)
    images2 = torch.stack(image2, 0)
    # print(images1.shape)

    max_sentence_num = max(sentence_num)
    max_word_num = max(max_word_num)

    targets = np.zeros((len(captions), max_sentence_num + 1, max_word_num))
    prob = np.zeros((len(captions), max_sentence_num + 1))

    for i, caption in enumerate(captions):
        for j, sentence in enumerate(caption):
            targets[i, j, :len(sentence)] = sentence[:]
            prob[i][j] = len(sentence) > 0

    return images1, images2, targets, prob, id


def get_loader(data_dir,
               split,
               vocabulary,
               transform,
               batch_size,
               s_max,
               n_max,
               shuffle=False):

    dataset = ChestXrayDataSet(data_dir=data_dir,
                               split=split,
                               vocabulary=vocabulary,
                               transforms=transform,
                               s_max=s_max,
                               n_max=n_max)

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              collate_fn=collate_fn)
    return data_loader