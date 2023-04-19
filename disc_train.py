# -*- coding: utf-8 -*-
import time
import random
import argparse
import pickle
import torch
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import os
import wordfreq
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.autograd import Variable
from utils.discriminator import *
from utils.models import *
# from utils.dataset import *
from utils.loss import *
from utils.logger import Logger
from utils.padding import TxtDataset
import utils.word_embedding as data_utils
from utils.disc_dataset import *


class DiscBase:
    def __init__(self, args):
        self.args = args
        self.min_val_loss = 10000000000
        self.min_train_loss = 10000000000
        self.params = None

        self._init_model_path()
        self.disc_model_dir = self._init_disc_model_dir()
        self.writer = self._init_writer()
        self.vocab, self.vocab_count = self._init_vocab()
        self.model_state_dict = self._load_model_state_dict()

        self.disc_model = self._init_disc_model()
        self.activation = nn.ReLU()
        self.ce_criterion = self._init_ce_criterion()
        self.mse_criterion = self._init_mse_criterion()
        self.bce_criterion = self._init_bce_criterion()
        self.batch_size = 4

        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler()  # 自动调整学习率
        self.logger = self._init_logger()
        self.writer.write("{}\n".format(self.args))

    def train(self):
        for epoch_id in range(self.start_epoch, self.args.epochs):  # 训练的轮
            true_loss, reward_t = self._epoch_train_on_true_data()
            fake_loss, reward_f = self._epoch_train_on_fake_data()
            train_loss = true_loss + fake_loss
            val_loss = 0.0

            if self.args.mode == 'train':
                self.scheduler.step(train_loss)  # 对lr进行调整
            else:
                self.scheduler.step(val_loss.cpu().data.numpy())
            print("[{} - Epoch {}] train_loss:{}   val_loss:{} \n".format(self._get_now(),
                                                        epoch_id,
                                                        train_loss,
                                                        val_loss,
                                                        self.optimizer.param_groups[0]['lr']))
            self._save_model(epoch_id,
                             train_loss)

            self._log(train_loss=train_loss,
                      lr=self.optimizer.param_groups[0]['lr'],
                      epoch=epoch_id)

    def _epoch_train(self):
        raise NotImplementedError

    def _epoch_val(self):
        raise NotImplementedError

    def _init_disc_model_dir(self):
        disc_model_dir = os.path.join(self.args.disc_model_path, self.args.saved_model_name)

        if not os.path.exists(disc_model_dir):
            os.makedirs(disc_model_dir)

        disc_model_dir = os.path.join(disc_model_dir)

        if not os.path.exists(disc_model_dir):
            os.makedirs(disc_model_dir)

        return disc_model_dir

    def _init_vocab(self):
        with open(self.args.vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        print("Vocabulary Size:{}\n".format(len(vocab)))

        return vocab, len(vocab)

    def _load_model_state_dict(self):
        self.start_epoch = 0
        try:
            model_state = torch.load(self.args.load_disc_model_path)
            self.start_epoch = model_state['epoch']
            print("[Load Model-{} Succeed!]\n".format(self.args.load_model_path))
            print("Load From Epoch {}\n".format(model_state['epoch']))
            return model_state
        except Exception as err:
            print("[Load Model Failed] {}\n".format(err))
            return None

    def _init_disc_model(self):
        model = Discriminator(seq_length=1,
                              vocab_size=self.vocab_count,
                              emb_size=32,
                              filter_size=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                              num_filter=[100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160],
                              dropoutRate=0.1)
        try:
            model = torch.load(self.args.load_disc_model_path)
            print("[Load Discriminator Succeed!]\n")
        except Exception as err:
            print("[Load Discriminator Model Failed] {}\n".format(err))

        if not self.args.disc_trained:
            for i, param in enumerate(model.parameters()):
                param.requires_grad = False
        else:
            if self.params:
                self.params += list(model.parameters())
            else:
                self.params = list(model.parameters())

        if self.args.cuda:
            model = model.cuda()

        return model

    def _init_data_loader(self):  # 加载数据  true data
        data_loader = get_loader(text_path=self.args.disc_train_true_data_list,
                                 vocabulary=self.vocab,
                                 batch_size=self.batch_size,
                                 s_max=6,
                                 n_max=30,
                                 shuffle=True)
        return data_loader

    def _init_data_loader_fake(self):  # 加载数据  fake data
        data_loader = get_loader(text_path=self.args.disc_train_fake_data_list,
                                 vocabulary=self.vocab,
                                 batch_size=self.batch_size,
                                 s_max=6,
                                 n_max=30,
                                 shuffle=True)
        return data_loader


    @staticmethod
    def softmax(x):
        prob = np.exp(x) / np.sum(np.exp(x), axis=0)
        return prob

    @staticmethod
    def _init_ce_criterion():
        return nn.CrossEntropyLoss(size_average=False, reduce=False)

    @staticmethod
    def _init_mse_criterion():
        return nn.MSELoss()

    @staticmethod
    def _init_bce_criterion():
        return nn.BCELoss()

    def _init_optimizer(self):
        return torch.optim.Adam(params=self.params, lr=self.args.learning_rate)

    def _log(self,
             train_loss,
             lr,
             epoch):
        info = {
            'train loss': train_loss,
            'learning rate': lr
        }

        for tag, value in info.items():
            self.logger.scalar_summary(tag, value, epoch + 1)

    def _init_logger(self):
        logger = Logger(os.path.join(self.disc_model_dir, 'logs'))
        return logger

    def _init_writer(self):
        writer = open(os.path.join(self.disc_model_dir, 'logs.txt'), 'w')
        return writer

    def _to_var(self, x, requires_grad=True):
        if self.args.cuda:
            x = x.cuda()
        return Variable(x, requires_grad=requires_grad)

    def _get_date(self):
        return str(time.strftime('%Y%m%d', time.gmtime()))

    def _get_now(self):
        return str(time.strftime('%Y%m%d-%H:%M', time.gmtime()))

    def _init_scheduler(self):
        scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=self.args.patience, factor=0.1)
        return scheduler

    def _init_model_path(self):
        if not os.path.exists(self.args.disc_model_path):
            os.makedirs(self.args.disc_model_path)

    def _init_log_path(self):
        if not os.path.exists(self.args.log_path):
            os.makedirs(self.args.log_path)

    def _save_model(self,
                    epoch_id,
                    train_loss):
        def save_whole_model(_filename):
            print("Saved Model in {}\n".format(_filename))
            torch.save({'discriminator': self.disc_model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'epoch': epoch_id},
                       os.path.join(self.disc_model_dir, "{}".format(_filename)))

        def save_part_model(_filename, value):
            print("Saved Model in {}\n".format(_filename))
            torch.save({"model": value},
                       os.path.join(self.disc_model_dir, "{}".format(_filename)))

        if train_loss < self.min_train_loss:
            file_name = "disc_train_best_loss.pth.tar"
            save_whole_model(file_name)
            self.min_train_loss = train_loss


class Debugger(DiscBase):
    def _init_(self, args):
        DiscBase.__init__(self, args)
        self.args = args

    def _epoch_train_on_true_data(self):
        true_loss = 0
        self.disc_model.train()
        train_data_loader = self._init_data_loader()
        for i, inputs in enumerate(train_data_loader):
            a = []
            print (inputs)
            labels = torch.LongTensor(np.ones([self.batch_size, 1], dtype=np.int64))
            labels = self._to_var(labels, requires_grad=False)
            inputs = self._to_var(torch.Tensor(inputs).float(), requires_grad=False)
            inputs = inputs.view(self.batch_size, -1)

            for j in range(1, inputs.shape[1]):
                output = self.disc_model.forward(inputs[:, j:j + 1].long())
            # for j in range(1, inputs.shape[1]):[:, j:j + 1]
            # output = self.disc_model.forward(inputs.long())
            # # out = output.view(self.batch_size, output.shape[0] // self.batch_size, output.shape[1])
            # #
            # # result = torch.zeros(self.batch_size, output.shape[1])
            #
            # # for i in range(out.shape[0]):
            # #     for j in range(out.shape[1]):
            # #         for k in range(out.shape[2]):
            # #             result[i][k] += (out[i][j][k] / out.shape[1])
                output = self._to_var(output, requires_grad=False)
                indices = torch.LongTensor([1])
                output = torch.index_select(output, 1, indices.cuda())
                # print("result", output)
                batch_loss = self.bce_criterion(output, labels.float()).sum()
                batch_loss = self._to_var(batch_loss, requires_grad=True)

                self.optimizer.zero_grad()
                batch_loss.backward()
                if self.args.clip > 0:
                    torch.nn.utils.clip_grad_norm(self.disc_model.parameters(), self.args.clip)
                self.optimizer.step()
                true_loss += batch_loss.item()
                # print("true loss:", true_loss)
            return true_loss, output

    def _epoch_train_on_fake_data(self):
        fake_loss = 0
        self.disc_model.train()
        train_data_loader = self._init_data_loader_fake()
        for i, inputs in enumerate(train_data_loader):
            a = []
            labels = torch.LongTensor(np.zeros([self.batch_size, 1], dtype=np.int64))
            labels = self._to_var(labels, requires_grad=False)
            inputs = self._to_var(torch.Tensor(inputs).float(), requires_grad=False)
            inputs = inputs.view(self.batch_size, -1)
            for j in range(1, inputs.shape[1]):
                output = self.disc_model.forward(inputs[:, j:j + 1].long())
            # out = output.view(self.batch_size, output.shape[0] // self.batch_size, output.shape[1])
            #
            # result = torch.zeros(self.batch_size, output.shape[1])

            # for i in range(out.shape[0]):
            #     for j in range(out.shape[1]):
            #         for k in range(out.shape[2]):
            #             result[i][k] += (out[i][j][k] / out.shape[1])
                output = self._to_var(output, requires_grad=False)
                indices = torch.LongTensor([0])
                output = torch.index_select(output, 1, indices.cuda())
                # print("result", output)
                batch_loss = self.bce_criterion(output, labels.float()).sum()
                batch_loss = self._to_var(batch_loss, requires_grad=True)
                self.optimizer.zero_grad()
                batch_loss.backward()
                if self.args.clip > 0:
                    torch.nn.utils.clip_grad_norm(self.disc_model.parameters(), self.args.clip)
                self.optimizer.step()
                fake_loss += batch_loss.item()
                # print("fake loss:", fake_loss)
            return fake_loss, output

    def _epoch_val(self):
        tag_loss, stop_loss, word_loss, loss = 0, 0, 0, 0
        # self.extractor.eval()
        # self.mlc.eval()
        # self.co_attention.eval()
        # self.sentence_model.eval()
        # self.word_model.eval()
        #
        # for i, (images, _, label, captions, prob) in enumerate(self.val_data_loader):
        #      batch_tag_loss, batch_stop_loss, batch_word_loss, batch_loss = 0, 0, 0, 0
        #      images = self._to_var(images, requires_grad=False)
        #
        #      visual_features, avg_features = self.extractor.forward(images)
        #      tags, semantic_features = self.mlc.forward(avg_features)
        #
        #      batch_tag_loss = self.mse_criterion(tags, self._to_var(label, requires_grad=False)).sum()
        #
        #      sentence_states = None
        #      prev_hidden_states = self._to_var(torch.zeros(images.shape[0], 1, self.args.hidden_size))
        #
        #      context = self._to_var(torch.Tensor(captions).long(), requires_grad=False)
        #      prob_real = self._to_var(torch.Tensor(prob).long(), requires_grad=False)
        #
        #      for sentence_index in range(captions.shape[1]):
        #          ctx, v_att, a_att = self.co_attention.forward(avg_features,
        #                                                        semantic_features,
        #                                                        prev_hidden_states)
        #
        #          topic, p_stop, hidden_states, sentence_states = self.sentence_model.forward(ctx,
        #                                                                                      prev_hidden_states,
        #                                                                                      sentence_states)
        #          print("p_stop:{}".format(p_stop.squeeze()))
        #          print("prob_real:{}".format(prob_real[:, sentence_index]))
        #
        #          batch_stop_loss += self.ce_criterion(p_stop.squeeze(), prob_real[:, sentence_index]).sum()
        #
        #          for word_index in range(1, captions.shape[2]):
        #              words = self.word_model.forward(topic, context[:, sentence_index, :word_index])
        #              word_mask = (context[:, sentence_index, word_index] > 0).float()
        #              batch_word_loss += (self.ce_criterion(words, context[:, sentence_index, word_index])
        #                                  * word_mask).sum()
        #              print("words:{}".format(torch.max(words, 1)[1]))
        #              print("real:{}".format(context[:, sentence_index, word_index]))
        #
        #      batch_loss = self.args.lambda_tag * batch_tag_loss \
        #                   + self.args.lambda_stop * batch_stop_loss \
        #                   + self.args.lambda_word * batch_word_loss
        #
        #      tag_loss += self.args.lambda_tag * batch_tag_loss.data
        #      stop_loss += self.args.lambda_stop * batch_stop_loss.data
        #      word_loss += self.args.lambda_word * batch_word_loss.data
        #      loss += batch_loss.data

        return tag_loss, stop_loss, word_loss, loss


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()

    """
    Data Argument
    """
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--mode', type=str, default='train')

    # Path Argument
    parser.add_argument('--vocab_path', type=str, default='./data/new_data/vocab.pkl',
                        help='the path for vocabulary object')
    parser.add_argument('--disc_train_true_data_list', type=str, default='./data/new_data/captions.txt',
                        help='the path for True data')
    parser.add_argument('--disc_train_fake_data_list', type=str, default='./data/new_data/disc_train_fake_data.txt',
                        help='the path for Fake data')

    parser.add_argument('--val_file_list', type=str, default='./data/new_data/val_data.txt',
                        help='the val array')

    # Load/Save model argument
    parser.add_argument('--disc_model_path', type=str, default='./report_disc_models/',
                        help='path for saving disc model')
    parser.add_argument('--disc_trained', action='store_true', default=True,
                        help='Whether train visual extractor or not')
    parser.add_argument('--load_disc_model_path', type=str, default='.',
                        help='The path of loaded disc model')
    parser.add_argument('--saved_model_name', type=str, default='v4',
                        help='The name of saved model')

    """
    Training Argument
    """
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=int, default=0.005)
    parser.add_argument('--epochs', type=int, default=100)  # 1000

    parser.add_argument('--clip', type=float, default=-1,
                        help='gradient clip, -1 means no clip (default: 0.35)')

    # Loss Function
    parser.add_argument('--lambda_tag', type=float, default=10000)
    parser.add_argument('--lambda_stop', type=float, default=10)
    parser.add_argument('--lambda_word', type=float, default=1)

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    debugger = Debugger(args)
    debugger.train()
