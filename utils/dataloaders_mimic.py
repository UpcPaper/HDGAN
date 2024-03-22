import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from .datasets_mimic import MimiccxrSingleImageDataset
from utils.build_vocab import Vocabulary, JsonReader

class R2DataLoader(DataLoader):
    def __init__(self, args,s_max, n_max,   vocabulary,file_list, tokenizer, split, transform):
        self.args = args
        self.s_max = s_max
        self.n_max = n_max
        self.vocab = vocabulary
        self.dataset_name = 'mimic_cxr'
        self.batch_size = args.batch_size
        self.shuffle = True
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
        self.file_list = file_list
        self.transform = transform
        # self.file_names, self.labels = self.__load_label_list(self.file_list)  # 图片名 和 label
        self.split = split
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((self.args.crop_size, self.args.crop_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((self.args.crop_size, self.args.crop_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        self.dataset = MimiccxrSingleImageDataset(self.args, self.s_max, self.n_max, self.vocab, self.file_list, self.tokenizer,
                                                  self.split, self.transform)

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers
        }
        super().__init__(**self.init_kwargs)

    # def __load_label_list(self, file_list):  # 加载标签
    #     print("datalaoder  ======执行load_label_list")
    #     labels = []
    #     filename_list = []
    #     with open(file_list, 'r') as f:
    #         for line in f:
    #             items = line.split()
    #             image_name = items[0]  # 图片名
    #             label = items[1:]
    #             label = [int(i) for i in label]
    #             image_name = '{}.png'.format(image_name)
    #             # image_name = '{}'.format(image_name)
    #             filename_list.append(image_name)
    #             labels.append(label)
    #     return filename_list, labels

    def collate_fn(self, data):
        images_id, images, label, captions, sentence_num, max_word_num = zip(*data)
        images = torch.stack(images, 0)

        max_sentence_num = max(sentence_num)  # 最大句子数
        max_word_num = max(max_word_num)  # 句子的最大单词数
        targets = np.zeros((len(captions), max_sentence_num, max_word_num + 1))
        prob = np.zeros((len(captions), max_sentence_num))

        for i, caption in enumerate(captions):
            for j, sentence in enumerate(caption):
                targets[i, j, :len(sentence)] = sentence[:]
                prob[i][j] = len(sentence) > 0
        return images, images_id, torch.Tensor(label), targets, prob

        # max_seq_length = max(seq_lengths)
        # targets = np.zeros((len(reports_ids), max_seq_length), dtype=int)
        # targets_masks = np.zeros((len(reports_ids), max_seq_length), dtype=int)
        # for i, report_ids in enumerate(reports_ids):
        #     targets[i, :len(report_ids)] = report_ids
        # for i, report_masks in enumerate(reports_masks):
        #     targets_masks[i, :len(report_masks)] = report_masks
        # return images_id, images,  torch.FloatTensor(label), torch.LongTensor(targets), torch.FloatTensor(targets_masks)

