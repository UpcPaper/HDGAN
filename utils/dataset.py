import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import numpy as np
from torchvision import transforms
import pickle
import matplotlib.pyplot as plt


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
        self.transform = transforms
        self.s_max = s_max
        self.n_max = n_max
        self.image1, self.image2, self.caption = self.__load_label_list(data_dir, split)

    def __load_label_list(self, data_dir, split):
        with open(data_dir, 'r') as f:
            data = json.load(f)
        data_all = data[split]

        image1 = []
        image2 = []
        labels = []

        for line in range(len(data_all)):
            image_name1 = data_all[line]['image_path'][0]
            image1.append(image_name1)
            image_name2 = data_all[line]['image_path'][1]
            image2.append(image_name2)
            labels.append(data_all[line]['report'])
        # print(image1.shape)
        return image1, image2, labels

    def __getitem__(self, index):
        image_1 = self.image1[index]
        image_2 = self.image2[index]
        id = image_1.split("_")[0]
        DATA_PATH = "/home/upc/sxx/Data/iu_xray/images/"

        image1 = Image.open(''.join([DATA_PATH, image_1])).convert('RGB')
        image2 = Image.open(''.join([DATA_PATH, image_2])).convert('RGB')

        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
            text = self.caption[index]
            text = text.lower()
            text = text.replace(',', '')
        # print(image2.shape)
        target = list()
        max_word_num = 0
        for i, sentence in enumerate(text.split('. ')):
            if i >= self.s_max:
                break
            sentence = sentence.replace('.', '')
            sentence = sentence.split()
            if len(sentence) == 0 or len(sentence) == 1 or len(sentence) > self.n_max-2:
                continue
            tokens = list()
            tokens.append(self.vocab('<start>'))
            tokens.extend([self.vocab(token) for token in sentence])
            tokens.append(self.vocab('<end>'))
            if max_word_num < len(tokens):
                max_word_num = len(tokens)
            target.append(tokens)
        sentence_num = len(target)
        return image1, image2, target, sentence_num, max_word_num, id

    def __len__(self):
        return len(self.image1)


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
                                              collate_fn=collate_fn,
                                              drop_last=True)
    return data_loader


if __name__ == '__main__':
    DATA_PATH = '/home/upc/sxx/Data/iu_xray'
    vocab_path = './vocab.pkl'
    data_path = DATA_PATH + '/annotation.json'
    split = 'train'
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    print("Vocab Size:{}\n".format(len(vocab)))
    batch_size = 2
    resize = 224
    crop_size = 224
    transform = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    data_loader = get_loader(data_dir=data_path,
                             split=split,
                             vocabulary=vocab,
                             transform=transform,
                             batch_size=batch_size,
                             s_max=7,
                             n_max=20,
                             shuffle=True)

    for i, (images1, images2, targets, prob, id) in enumerate(data_loader):
        # print(images1.shape)  # torch.Size([BS, 3, 224, 224])
        # print(images2.shape)  # torch.Size([BS, 3, 224, 224])
        plt.imshow(images1[0][0])
        plt.show()
        plt.imshow(images2[0][0])
        plt.show()
        plt.imshow(images1[1][0])
        plt.show()
        plt.imshow(images2[1][0])
        plt.show()
        print(targets)
        print(prob)
        print(id)
        break
