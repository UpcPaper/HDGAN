# -*- coding: utf-8 -*-
import time
import os
os.environ['CUDA_LAUNCH_BLOCKING']="1"
import random
import sys
import traceback
import threading
import argparse
import pickle
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, TensorDataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, TensorDataset
from utils.tokenizers import Tokenizer
# from utils.dataloaders_mimic import R2DataLoader
# from utils.dataloaders_mimic import R2DataLoader
# from utils.datasets_mimic import *

from utils.models import *
from utils.discriminator import *
from utils.discriminators import *

from tqdm import tqdm
from utils.tokenizers import Tokenizer

from utils.disc_dataset import *
from utils.dataset_cov import get_loader as get_loader_t
from adver_trainer import LSTMDebugger
from metric_performance import compute_scores


class AdversarialBase:
    def __init__(self, args):
        self.args = args
        self.params_d = None
        self.params_d_1 = None

        self.params = None
        self.train_loss = 0
        self.batch_size = self.args.batch_size
        self.max_bleu1 = -0.001
        self.g_min_train_loss = 100000000000
        self.d_min_train_loss = 100000000000
        self.d_min_train_loss_1 = 100000000000

        self._init_model_path()
        self.vocab, self.vocab_count = self._init_vocab()

        self.extractor = self._init_visual_extractor()
        # self.mlc = self._init_mlc()
        # self.co_attention = self._init_co_attention()
        self.semantic = self._init_semantic_embedding()
        self.sentence_model = self._init_sentence_model()
        self.word_model = self._init_word_model()
        self.model_state_dict = self._load_model_state_dict()
        self.disc_model = self._init_disc_model()
        self.discs_model = self._init_discs_model()
        self.bce_criterion = self._init_bce_criterion()
        self.ce_criterion = self._init_ce_criterion()
        self.mse_criterion = self._init_mse_criterion()

        self.reward = torch.zeros(self.args.batch_size, 1)
        self.optimizer = self._init_optimizer()
        self.optimizer_d = self._init_optimizer_d()
        self.optimizer_d_1 = self._init_optimizer_d_1()

        self.model_dir_g = self._init_model_dir_g()
        self.model_dir = self._init_model_dir()

        self.gen_model = torch.load(self.args.load_model_path)
        # self.logger = self._init_logger()
        # self.tokenizer = Tokenizer(args)
        self.file_list = self.args.file_list
        self.split = 'test'
        self.s_max = self.args.s_max
        self.n_max = self.args.n_max
        self.transform = transforms.Compose([
            transforms.Resize(300),
            transforms.RandomCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        # self.data_loader = self._init_data_loader(self.args.adver_file_list, self.transform)
        self.data_loader = self._init_data_loader(split='train', transform=self.transform, shuffle=True)
        # self.data_loader = R2DataLoader(args, s_max=self.s_max,  n_max=self.n_max, vocabulary=self.vocab,
        #                                       file_list=self.file_list,  tokenizer=self.tokenizer,  split=self.split,
        #                                       transform=self.transform)
        self.test_data_loader = self._init_data_loader(split='test', transform=self.transform, shuffle=False)

    def _init_data_loader(self, split, transform, shuffle):
        data_loader = get_loader_t(data_dir=self.args.data_dir,
                                 split=split,
                                 vocabulary=self.vocab,
                                 transform=transform,
                                 batch_size=self.args.batch_size,
                                 s_max=self.args.s_max,
                                 n_max=self.args.n_max,
                                 shuffle=shuffle)
        return data_loader

    def _init_vocab(self):
        with open(self.args.vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        # print("Vocabulary Size:{}\n".format(len(vocab)))
        return vocab, len(vocab)
    def _init_model_path(self):
        if not os.path.exists(self.args.model_path):
            os.makedirs(self.args.model_path)

    def _init_data_loader_true(self):  # 加载数据  true data

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


    def _load_model_state_dict(self):
        self.start_epoch = 0
        try:
            model_state = torch.load(self.args.load_disc_model_path)
            self.start_epoch = model_state['epoch']
            print("[Load Discriminator -{} Succeed!]\n".format(self.args.load_disc_model_path))
            print("Load From Epoch {}\n".format(model_state['epoch']))
            return model_state
        except Exception as err:
            print("[Load Discriminator Failed] {}\n".format(err))
            return None

    def _init_visual_extractor(self):
        model = VisualFeatureExtractor(self.args.embed_size)

        try:
            model_state = torch.load(self.args.load_visual_model_path)
            model.load_state_dict(model_state['extractor'])
            print("[Load Visual Extractor Succeed!]\n")
        except Exception as err:
            print("[Load Visual Extractor Model Failed] {}\n".format(err))

        if not self.args.visual_trained:
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

    def _init_semantic_embedding(self):
        model = SemanticEmbedding(embed_size=self.args.embed_size)
        try:
            model_state = torch.load(self.args.load_semantic_model_path)
            model.load_state_dict(model_state['semantic'])
            print("[Load Semantic Embedding Succeed!]\n")
        except Exception as err:
            print("[Load Semantic Embedding Failed {}!]\n".format(err))
        if not self.args.semantic_trained:
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

    def _init_mlc(self):
        model = MLC(classes=self.args.classes,
                    sementic_features_dim=self.args.sementic_features_dim,
                    fc_in_features=self.extractor.out_features,
                    k=self.args.k)

        try:
            model_state = torch.load(self.args.load_mlc_model_path)
            model.load_state_dict(model_state['mlc'])
            # print("[Load MLC Succeed!]\n")
        except Exception as err:
            print("[Load MLC Failed {}!]\n".format(err))

        if not self.args.mlc_trained:
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

    def _init_co_attention(self):
        model = CoAttention(version=self.args.attention_version,
                            embed_size=self.args.embed_size,
                            hidden_size=self.args.hidden_size,
                            visual_size=self.extractor.out_features,
                            k=self.args.k,
                            momentum=self.args.momentum)

        try:
            model_state = torch.load(self.args.load_co_model_path)
            model.load_state_dict(model_state['co_attention'])
            # print("[Load Co-attention Succeed!]\n")
        except Exception as err:
            print("[Load Co-attention Failed {}!]\n".format(err))

        if not self.args.co_trained:
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

    def _init_sentence_model(self):
        raise NotImplementedError

    def _init_word_model(self):
        raise NotImplementedError

    # def _init_logger(self):
    #     logger = open('./results/results.txt', 'w')
    #     return logger

    def __save_json(self, result):
        result_path = self.args.result_path
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        with open(os.path.join(result_path, '{}.json'.format(self.args.result_name)), 'w') as f:
            json.dump(result, f)  # 将json信息写进文件  dump

    def __save_json_test(self, result):
        result_path = self.args.result_path_test
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        with open(os.path.join(result_path, '{}.json'.format(self.args.result_name)), 'w') as f:
            json.dump(result, f)  # 将json信息写进文件  dump



    def __vec2sent(self, words_id):  # array是word_id 将Word_id转成单词
        sampled_caption = []
        for word_id in words_id:
            word = self.vocab.get_word_by_id(word_id)
            if word == '<start>':
                continue
            if word == '<end>' or word == '<pad>':
                break
            sampled_caption.append(word)
        return ' '.join(sampled_caption)

    def generate(self):
        self.extractor.train()
        self.semantic.train()
        self.sentence_model.train()
        self.word_model.train()
        progress_bar = tqdm(self.data_loader, desc='Generating')
        results = {}
        with open("./data/new_data/all_true_data_cov.json", 'r') as fj:
            data = json.load(fj)
        write1 = open('./data/new_data/disc_train_fake_data.txt', 'w')
        fj = open('./data/new_data/disc_train_true_data.txt', 'w')
        for images1, images2, captions, prob, image_id in progress_bar:
            images_frontal = self._to_var(images1, requires_grad=False)
            images_lateral = self._to_var(images2, requires_grad=False)
            frontal, lateral, avg = self.extractor.forward(images_frontal, images_lateral)
            state_c, state_h = self.semantic.forward(avg)
            state = (torch.unsqueeze(state_c, 0), torch.unsqueeze(state_h, 0))
            pre_hid = torch.unsqueeze(state_h, 1)
            # tags, semantic_features = self.mlc.forward(avg_features)
            # sentence_states = None
            # prev_hidden_states = self._to_var(torch.zeros(images.shape[0], 1, self.args.hidden_size))
            pred_sentences = {}  # 预测
            real_sentences = {}  # 真实
            for i in image_id:
                pred_sentences[i] = {}  # 具体到每一张
                real_sentences[i] = {}
            for i in range(self.args.s_max):  # 句子数
                # ctx, alpha_v, alpha_a = self.co_attention.forward(avg_features, semantic_features, prev_hidden_states)
                topic,  p_stop, state, h0_word, c0_word, pre_hid = self.sentence_model.forward(frontal, lateral, state, pre_hid)
                p_stop = p_stop.squeeze(1)
                p_stop = torch.unsqueeze(torch.max(p_stop, 1)[1], 1)
                start_tokens = np.zeros(images_frontal.shape[0])
                state_word =(c0_word, h0_word)
                start_tokens[:] = self.vocab('<start>')
                start_tokens = self._to_var(torch.Tensor(start_tokens).long(), requires_grad=False)
                sample_ids,_ = self.word_model.sample(start_tokens, state_word)
                sample_ids = sample_ids * p_stop.cpu().numpy()
                # prev_hidden_states = hidden_state
                for id, words_id in zip(image_id, sample_ids):
                    pred_sentences[id][i] = self.__vec2sent(words_id)  # cpu().detach().numpy()
            for id, array in zip(image_id, captions):
                for i, sent in enumerate(array):
                    real_sentences[id][i] = self.__vec2sent(sent)
            for id in image_id:
                results[id] = {
                    'Pred Sent': pred_sentences[id],
                    'Real Sent': real_sentences[id]
                }
                write1.write(str(pred_sentences[id]) + "." + "\n")
                fj.write(data[str(id)]+"\n")
        fj.close()
        write1.close()
        #  操作 disc_fake
        with open('./data/new_data/disc_train_fake_data.txt', 'r') as fr:
            lines = fr.readlines()
            for i, line in enumerate(lines):
                lines[i] = str(lines[i]).replace('{', '').replace('}', '')  # 去除[],这两行按数据不同，可以选择
                lines[i] = str(lines[i]).replace('0:', '').replace('1:', '').replace('2:', '').replace('3:', '').replace(
                        '4:', '').replace('5:', '')
                lines[i] = str(lines[i]).replace("'", '')  # 去除单引号,每行末尾追加换行符
                lines[i] = str(lines[i]).replace(", ", '.')
        f = open('./data/new_data/disc_train_fake_data.txt', 'w')
        f.writelines(lines)
        f.close()
        self.__save_json(results)

    def test(self):
        self.extractor.train()
        # self.mlc.train()
        # self.co_attention.train()
        self.semantic.train()
        self.sentence_model.train()
        self.word_model.train()

        progress_bar = tqdm(self.test_data_loader, desc='Generating')
        results = {}
        for images1, images2, captions, prob, image_id in progress_bar:
            images_frontal = self._to_var(images1, requires_grad=False)
            images_lateral = self._to_var(images2, requires_grad=False)
            # visual_features, avg_features = self.extractor.forward(images)
            # tags, semantic_features = self.mlc.forward(avg_features)
            # print(str(images_frontal.shape) + " " + str(images_lateral.shape))
            frontal, lateral, avg = self.extractor.forward(images_frontal, images_lateral)
            state_c, state_h = self.semantic.forward(avg)
            state = (torch.unsqueeze(state_c, 0), torch.unsqueeze(state_h, 0))
            pre_hid = torch.unsqueeze(state_h, 1)

            # sentence_states = None
            # prev_hidden_states = self.__to_var(torch.zeros(images.shape[0], 1, self.args.hidden_size))
            pred_sentences = {}  # 预测
            real_sentences = {}  # 真实
            for i in image_id:
                pred_sentences[i] = {}  # 具体到每一张
                real_sentences[i] = {}

            for i in range(self.args.s_max):   # 句子数
                # ctx, alpha_v, alpha_a = self.co_attention.forward(avg_features, semantic_features, prev_hidden_states)
                # topic, p_stop, hidden_state, sentence_states = self.sentence_model.forward(ctx,
                #                                                                            prev_hidden_states,
                #                                                                            sentence_states)
                _, p_stop, state, h0_word, c0_word, pre_hid = self.sentence_model.forward(frontal, lateral, state,
                                                                                          pre_hid)
                p_stop = p_stop.squeeze(1)
                p_stop = torch.unsqueeze(torch.max(p_stop, 1)[1], 1)
                start_tokens = np.zeros(images_frontal.shape[0])  # [4, 1]
                # start_tokens[:, 0] = self.vocab('<start>')
                # start_tokens = self.__to_var(torch.Tensor(start_tokens).long(), requires_grad=False)
                #
                # sample_ids = self.word_model.sample(topic, start_tokens)
                state_word = (c0_word, h0_word)
                start_tokens[:] = self.vocab('<start>')
                start_tokens = self._to_var(torch.Tensor(start_tokens).long(), requires_grad=False)
                sample_ids,_ = self.word_model.sample(start_tokens, state_word)  # [4,50]
                sample_ids = sample_ids * p_stop.cpu().numpy()
                # p_stop = p_stop.squeeze(1)
                # p_stop = torch.max(p_stop, 1)[1].unsqueeze(1)
                # sample_ids = torch.Tensor(sample_ids).cpu() * p_stop.cpu()

                # prev_hidden_states = hidden_state
                # print(type(sample_ids))
                for id, array in zip(image_id, sample_ids):
                    pred_sentences[id][i] = self.__vec2sent(array)  # cpu().detach().numpy()
                # sys.exit()
            # for i in image_id:
            #     with open("./data/new_data/all_true_data.json", 'r') as fj:
            #         data = json.load(fj)
            #     fj = open('./data/new_data/disc_train_true_data.txt', 'a')
            #     fj.writelines(data[i]+"\n")
            for id, array in zip(image_id, captions):
                for i, sent in enumerate(array):
                    real_sentences[id][i] = self.__vec2sent(sent)
            for id in image_id:
                results[id] = {
                    # 'Real Tags': self.tagger.inv_tags2array(real_tag),
                    # 'Pred Tags': self.tagger.array2tags(torch.topk(pred_tag, self.args.k)[1].cpu().data.numpy()),
                    'Pred Sent': pred_sentences[id],
                    'Real Sent': real_sentences[id]
                }
                # print(id)
                # print("pred_sentences", pred_sentences[id])
                # print("=====================================================")

        self.__save_json_test(results)
        gts = []
        res = []
        for key in results:
            gt = ""
            re = ""
            for i in results[key]["Real Sent"]:
                if results[key]["Real Sent"][i] != "":
                    gt = gt + results[key]["Real Sent"][i] + " . "

            for i in results[key]["Pred Sent"]:
                if results[key]["Pred Sent"][i] != "":
                    re = re + results[key]["Pred Sent"][i] + " . "
            gts.append(gt)
            res.append(re)

        test_met = compute_scores({i: [gt] for i, gt in enumerate(gts)},
                                  {i: [re] for i, re in enumerate(res)})
        print(test_met)
        return test_met

    @staticmethod
    def _init_mse_criterion():
        return nn.MSELoss()
    @staticmethod
    def _init_bce_criterion():
        return nn.BCELoss()

    @staticmethod
    def _init_ce_criterion():
        return nn.CrossEntropyLoss(size_average=False, reduce=False)

    def _init_optimizer(self):
        return torch.optim.Adam(params=self.params, lr=self.args.learning_rate)
    def _init_optimizer_d(self):  # 判别器优化
        return torch.optim.Adam(params=self.params_d, lr=self.args.learning_rate)

    def _init_optimizer_d_1(self):  # 判别器_1优化
        return torch.optim.Adam(params=self.params_d_1, lr=self.args.learning_rate)

    def _init_model_dir(self):
        model_dir = os.path.join(self.args.load_disc_model_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_dir = os.path.join(model_dir)
        return model_dir

    def _init_model_dir_1(self):
        model_dir = os.path.join(self.args.load_discs_model_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_dir = os.path.join(model_dir)
        return model_dir
    def _init_model_dir_g(self):
        model_dir = os.path.join(self.args.load_model_path)

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_dir = os.path.join(model_dir)
        return model_dir
    def _init_disc_model(self):  # 加载判别器
        model = Discriminator(seq_length=1,
                              vocab_size=self.vocab_count,
                              emb_size=32,
                              filter_size=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                              num_filter=[100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160],
                              dropoutRate=0.1)
        try:
            model_state = torch.load(self.args.load_disc_model_path)
            model.load_state_dict(model_state['discriminator'])
            print("[Load Discriminator Succeed!]\n")
        except Exception as err:
            print("[Load Discriminator Model Failed] {}\n".format(err))

        if not self.args.disc_trained:
            for i, param in enumerate(model.parameters()):
                param.requires_grad = False
        else:
            if self.params_d:
                self.params_d += list(model.parameters())
            else:
                self.params_d = list(model.parameters())
        if self.args.cuda:
            model = model.cuda()
        return model

    def _init_discs_model(self):  # 加载判别器
        model = Discriminators(vocab_size=self.vocab_count,
                              input_size=50,
                              hidden_size=512,
                              num_class=2,
                              num_layers=1)
        try:
            model_state = torch.load(self.args.load_discs_model_path)
            model.load_state_dict(model_state['discs_model'])
            print("[Load Discriminators Succeed!]\n")
        except Exception as err:
            print("[Load Discriminators Model Failed] {}\n".format(err))

        if not self.args.disc_trained:
            for i, param in enumerate(model.parameters()):
                param.requires_grad = False
        else:
            if self.params_d_1:
                self.params_d_1 += list(model.parameters())
            else:
                self.params_d_1 = list(model.parameters())
        if self.args.cuda:
            model = model.cuda()
        return model
    def _save_model_g(self,
                    epoch_id,
                    g_loss, b1, b2, b3, b4, ROUGE_L, CIDEr, METEOR):
        def save_whole_model(_filename):
            print("Saved Model in {}\n".format(_filename))
            torch.save({'extractor': self.extractor.state_dict(),
                        'semantic': self.semantic.state_dict(),
                        'sentence_model': self.sentence_model.state_dict(),
                        'word_model': self.word_model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'epoch': epoch_id},
                       os.path.join(self.args.saved_model_name, "{}".format(_filename)))
        if g_loss < self.g_min_train_loss:
            file_name = "ad_train_best_loss.pth.tar"
            save_whole_model(file_name)
            self.g_min_train_loss = g_loss
        if b1 > self.max_bleu1:
            file_name = "test_best.pth.tar"
            save_whole_model(file_name)
            self.max_bleu1 = b1
            self.max_bleu2 = b2
            self.max_bleu3 = b3
            self.max_bleu4 = b4
            self.max_METEOR = METEOR
            self.max_ROUGE_L = ROUGE_L
            self.max_CIDEr = CIDEr

    def _save_model(self,
                    epoch_id,
                    loss):
        def save_whole_model(_filename):
            print("Saved Model in {}\n".format(_filename))
            torch.save({'discriminator': self.disc_model.state_dict(),
                        'optimizer': self.optimizer_d.state_dict(),
                        'epoch': epoch_id},
                       os.path.join(self.args.disc_saved_model_name, "{}".format(_filename)))

        if loss < self.d_min_train_loss:
            file_name = "ad_disc_train_best_loss.pth.tar"
            save_whole_model(file_name)
            self.d_min_train_loss = loss

    def _save_model_1(self,
                    epoch_id,
                    loss):
        def save_whole_model(_filename):
            print("Saved Model in {}\n".format(_filename))
            torch.save({'discs_model': self.discs_model.state_dict(),
                        'optimizer': self.optimizer_d_1.state_dict(),
                        'epoch': epoch_id},
                       os.path.join(self.args.discs_saved_model_name, "{}".format(_filename)))

        if loss < self.d_min_train_loss_1:
            file_name = "ad_discs_train_best_loss.pth.tar"
            save_whole_model(file_name)
            self.d_min_train_loss_1 = loss
    def _to_var(self, x, requires_grad=True):
        if self.args.cuda:
            x = x.cuda()
        return Variable(x, requires_grad=requires_grad)

    def _get_date(self):
        return str(time.strftime('%Y%m%d', time.gmtime()))

    def _get_now(self):
        return str(time.strftime('%Y%m%d-%H:%M', time.gmtime()))

    def loss_with_reward(self, prediction, x, rewards):
        embedding = nn.Embedding(213, 213)
        prediction = embedding(prediction.long())
        x1 = x.contiguous().view([-1, 1]).long()
        one_hot = torch.Tensor(x1.shape[0], 213).cuda()
        one_hot.zero_()
        x2 = one_hot.scatter_(1, x1, 1)
        pred1 = prediction.view([-1, 213])
        pred2 = torch.log(torch.clamp(pred1, min=1e-20, max=1.0))
        # print(str(prediction.shape) + " " + str(x2.shape) + " " + str(pred2.shape))
        prod = torch.mul(x2.cuda(), pred2.cuda())
        reduced_prod = torch.sum(prod, dim=1)
        rewards_prod = torch.mul(reduced_prod.cuda(), rewards.view([-1]).cuda())
        generator_loss = torch.sum(rewards_prod)
        return -generator_loss

    def loss_with_reward_1(self, prediction, x, rewards):
        embedding = nn.Embedding(213, 213)
        prediction = embedding(prediction.long())
        # print ("prediction after embedding:", prediction.shape)
        x1 = x.contiguous().view([-1, 1]).long()
        # print ("x1:", x1.shape)

        one_hot = torch.Tensor(x1.shape[0], 213).cuda()
        one_hot.zero_()
        x2 = one_hot.scatter_(1, x1, 1)

        pred1 = prediction.view([-1, 213])
        pred2 = torch.log(torch.clamp(pred1, min=1e-20, max=1.0))
        prod = torch.mul(x2.cuda(), pred2.cuda())
        reduced_prod = torch.sum(prod, dim=1)
        rewards_prod = torch.mul(reduced_prod.cuda(), rewards.view([-1]).cuda())
        generator_loss = torch.sum(rewards_prod)
        return -generator_loss

    def adver(self):
        print ("update generator")
        tag_loss, stop_loss, word_loss, loss = 0, 0, 0, 0
        for i, (images1, images2, captions, prob, image_id) in enumerate(self.data_loader):
            # print(i)
            batch_tag_loss, batch_stop_loss, batch_word_loss, batch_sentence_loss, batch_sentence_loss_1,batch_loss = 0, 0, 0, 0, 0, 0
            images_frontal = self._to_var(images1, requires_grad=False)
            images_lateral = self._to_var(images2, requires_grad=False)
            frontal, lateral, avg = self.extractor.forward(images_frontal, images_lateral)
            state_c, state_h = self.semantic.forward(avg)
            state = (torch.unsqueeze(state_c, 0), torch.unsqueeze(state_h, 0))
            pre_hid = torch.unsqueeze(state_h, 1)

            # 中间层
            # prev_hidden_states = self._to_var(torch.zeros(images.shape[0], 1, self.args.hidden_size), requires_grad=False)
            prob_real = self._to_var(torch.Tensor(prob).long(), requires_grad=False)
            captions = self._to_var(torch.Tensor(captions).long(), requires_grad=False)
            pred_sentences = []  # 预测
            reward_1 = []
            for sentence_index in range(0, captions.shape[1]):  # 是s_max=6 一个caption里有六句话

                # ctx, alpha_v, alpha_a = self.co_attention.forward(avg_features,
                #                                                   semantic_features,
                #                                                   prev_hidden_states)
                # topic, p_stop, hidden_states, sentence_states = self.sentence_model.forward(ctx,
                #                                                                             prev_hidden_states,
                #                                                                             sentence_states)
                # batch_stop_loss += self.ce_criterion(p_stop.squeeze(), prob_real[:, sentence_index]).sum().item()
                # for word_index in range(0, captions.shape[2]):  # 0
                #     words = self.word_model.forward(topic, context[:, sentence_index, :word_index])
                #     word_mask = (context[:, sentence_index, word_index] > 0).float()
                    # batch_word_loss += (self.ce_criterion(words, context[:, sentence_index, word_index])
                    #                     * word_mask).sum() * (0.9 ** word_index)
                topic, p_stop, state, h0_word, c0_word, pre_hid = self.sentence_model.forward(frontal, lateral, state, pre_hid)
                p_stop = p_stop.squeeze(1)
                p_stop = torch.unsqueeze(torch.max(p_stop, 1)[1], 1)
                start_tokens = np.zeros(images_frontal.shape[0])

                state_word = (c0_word, h0_word)
                start_tokens[:] = self.vocab('<start>')
                start_tokens = self._to_var(torch.Tensor(start_tokens).long(), requires_grad=False)
                sample_ids, _ = self.word_model.sample(start_tokens, state_word)  # [4,50]
                sample_ids = sample_ids * p_stop.cpu().numpy()
                reward = []
                # print(type(sample_ids))
                sample_ids = torch.from_numpy(sample_ids)
                # print(type(sample_ids))
                pred_sentences.append(sample_ids)
                for j in range(0, sample_ids.shape[1]):
                    output = self.disc_model.forward(sample_ids[:, j:j + 1].cuda().long())
                    output = self._to_var(output, requires_grad=False)
                    indices = torch.LongTensor([0])
                    out = torch.index_select(output, 1, indices.cuda())
                    for i in out:
                        reward.append(i.item())
                reward = np.transpose(np.array(reward)) / 1.0
                reward = torch.Tensor(reward)
                s = []
                a = [0]
                # print(captions.shape)
                for i in range(0, captions[:, sentence_index, :].shape[0]):
                    t = captions[:, sentence_index, :][i].tolist()  # 将tensor转为list
                    for j in range(0, self.args.n_max-captions[:, sentence_index, :].shape[1]):
                        t.extend(a)
                    s.append(t)
                context1 = torch.Tensor(s)
                # print(str(captions[:, sentence_index, :].shape) +" " + str(context1.shape))
                # sys.exit()
                # 每个词相加得到reward “语义奖励”
                batch_sentence_loss += (self.loss_with_reward(sample_ids, context1.cuda(), reward)).sum().item()
                # print("---------------")
            # 整个句子得到一个reward， “结构奖励”
            t = torch.LongTensor()
            for i in pred_sentences:
                pred_sentences_1 = np.asarray(i)
                pred_sentences_2 = torch.from_numpy(pred_sentences_1)
                t = torch.cat((t,pred_sentences_2.long()), 1)
            final_out = self.discs_model.forward(t)
            out_1 = torch.index_select(final_out, 1, indices)
            for i in out_1:
                reward_1.append(i.item())
            reward_1 = np.transpose(np.array(reward)) / 1.0
            reward_1 = torch.Tensor(reward_1)
            batch_sentence_loss_1 += (self.loss_with_reward_1(sample_ids, context1.cuda(), reward_1)).sum().item()
            # batch_sentence_loss = self._to_var(torch.tensor(batch_sentence_loss))
            # batch_sentence_loss_1 = self._to_var(torch.tensor(batch_sentence_loss_1))
            # batch_loss = self.args.lambda_tag * batch_tag_loss \
            #              + self.args.lambda_stop * batch_stop_loss \
            #              + self.args.lambda_word * batch_word_loss\
            #              + self.args.lambda_sentence * batch_sentence_loss \
            #              + self.args.lambda_sentence * batch_sentence_loss_1
            batch_loss = self.args.lambda_sentence * batch_sentence_loss \
                         + self.args.lambda_sentence * batch_sentence_loss_1
            batch_loss = self._to_var(torch.tensor(batch_loss))
            self.optimizer.zero_grad()  # 把梯度置零，也就是把loss关于weight的导数变成0
            batch_loss.backward()  # 反向传播求梯度retain_graph=True
            if self.args.clip > 0:
                # 最简单粗暴的方法，设定阈值，当梯度小于/大于阈值时，更新的梯度为阈值  梯度裁剪
                torch.nn.utils.clip_grad_norm(self.sentence_model.parameters(), self.args.clip)
                torch.nn.utils.clip_grad_norm(self.word_model.parameters(), self.args.clip)
            self.optimizer.step()  # 更新所有参数
            loss += batch_loss.item()  # 根本原因
        return loss

class Adversarial(AdversarialBase):
    def _init_(self, args):
        AdversarialBase.__init__(self, args)
        self.args = args
    def epoch_train(self):
        print('===Start Adversarial Training===')
        # Train the generator for one step
        # for it in range(1):  # 这里用的数据是生成器生成的假数据 通过判别器进行判断生成reward 来算生成器的损失（带有reward）,用来更新生成器
        #     train_data_loader = self._init_data_loader_fake()
        #
        #     for i, inputs in enumerate(train_data_loader):
        #         inputs = self._to_var(torch.Tensor(inputs).float(), requires_grad=False)
        #         print("inputs shape", inputs.shape)
        #         inputs = inputs.view(self.batch_size, -1)
        #         reward = []
        #         for j in range(inputs.shape[1]):
        #             output = self.disc_model.forward(inputs[:, j:j+1].long())
        #             output = self._to_var(output, requires_grad=False)
        #             indices = torch.LongTensor([0])
        #             out = torch.index_select(output, 1, indices.cuda())
        #             for i in out:
        #                 reward.append(i.item())
        #         reward = np.transpose(np.array(reward)) / 1.0
        for _ in range(1):
            g_loss = self.adver() # g_step

        # Test
        for _ in range(2):  # t
            print("Use New Generator To Generate Fake Data")
            self.generate()
            print("Train the discriminator")
            # 1A Train D on real
            for _ in range(3): # d_step
                d_loss_t, d_loss_f = 0.0, 0.0
                d_loss_t_1, d_loss_f_1 = 0.0, 0.0
                train_data_loader_t = self._init_data_loader_true()
                print("---------")
                for i, inputs in enumerate(train_data_loader_t):
                    batch_loss_t,batch_loss_t_1 = 0.0, 0.0
                    labels = torch.LongTensor(np.ones([self.batch_size, 1], dtype=np.int64))
                    labels = self._to_var(labels, requires_grad=False)
                    inputs = self._to_var(torch.Tensor(inputs).float(), requires_grad=False)
                    inputs = inputs.view(self.batch_size, -1)
                    # outputs = self.discs_model.forward(inputs.long())
                    # batch_loss_t_1 = self.ce_criterion(outputs.squeeze(), labels.squeeze().cpu()).sum()

                    for j in range(1, inputs.shape[1]):
                        output = self.disc_model.forward(inputs[:, j:j + 1].long())
                        output = self._to_var(output, requires_grad=False)
                        indices = torch.LongTensor([1])
                        output = torch.index_select(output, 1, indices.cuda())
                        batch_loss_t += self.bce_criterion(output, labels.float()).sum()
                    self.optimizer_d.zero_grad()
                    # self.optimizer_d_1.zero_grad()

                    batch_loss_t = self._to_var(batch_loss_t, requires_grad=True)
                    batch_loss_t.backward()
                    # batch_loss_t_1 = self._to_var(batch_loss_t_1, requires_grad=True)
                    # batch_loss_t_1.backward()
                    d_loss_t = batch_loss_t.item()
                    # d_loss_t_1 = batch_loss_t_1.item()

                # train on fake data
                print("********")
                train_data_loader_f = self._init_data_loader_fake()
                for i, inputs in enumerate(train_data_loader_f):
                    batch_loss_f = 0.0
                    labels = torch.LongTensor(np.zeros([self.args.batch_size, 1], dtype=np.int64))
                    labels = self._to_var(labels, requires_grad=False)
                    inputs = self._to_var(torch.Tensor(inputs).float(), requires_grad=False)
                    inputs = inputs.view(self.args.batch_size, -1)
                    # outputs = self.discs_model.forward(inputs.long())
                    # batch_loss_f_1 = self.ce_criterion(outputs.squeeze(), labels.squeeze().cpu()).sum()
                    for j in range(1, inputs.shape[1]):
                        output = self.disc_model.forward(inputs[:, j:j + 1].long())
                        output = self._to_var(output, requires_grad=False)
                        indices = torch.LongTensor([0])
                        output = torch.index_select(output, 1, indices.cuda())
                        batch_loss_f += (self.bce_criterion(output, labels.float())).sum()
                    # batch_loss_f = self._to_var(batch_loss_f, requires_grad=True)
                    # batch_loss_f_1 = self._to_var(batch_loss_f_1, requires_grad=True)
                    self.optimizer_d.zero_grad()
                    # self.optimizer_d_1.zero_grad()
                    batch_loss_f = self._to_var(torch.tensor(batch_loss_f))
                    # batch_loss_f_1 = self._to_var(torch.tensor(batch_loss_f_1))

                    batch_loss_f.backward()
                    # batch_loss_f_1.backward()
                    if self.args.clip > 0:
                        torch.nn.utils.clip_grad_norm(self.disc_model.parameters(), self.args.clip)
                    self.optimizer_d.step()
                    # self.optimizer_d_1.step()

                    d_loss_f = batch_loss_f.item()
                    # d_loss_f_1 = batch_loss_f_1.item()
                print("%%%%%%%")
        test_met = self.test()
        return g_loss, d_loss_t + d_loss_f, d_loss_t_1 + d_loss_f_1, test_met

    def train(self):
        Loss_list = []
        for epoch in range(0, self.args.epochs):
            print ("=======Epoch:", epoch, "======")
            g_loss, d_loss, d_loss_1, test_met = self.epoch_train()
            print(" D-train loss_t:{} -  lr:{}\n".format(d_loss,
                                                       self.optimizer_d.param_groups[0]['lr']))
            print(" D-train_1 loss_t:{} -  lr:{}\n".format(d_loss_1,
                                                         self.optimizer_d_1.param_groups[0]['lr']))
            print(" G-train loss:{} -  lr:{}\n".format(g_loss,
                                                       self.optimizer.param_groups[0]['lr']))
            # Loss_list.append(g_loss/1000000)
            # self.logger.write(str(g_loss/1000000) +  "\n")
            b1 = test_met['BLEU_1']
            b2 = test_met['BLEU_2']
            b3 = test_met['BLEU_3']
            b4 = test_met['BLEU_4']
            METEOR = test_met['METEOR']
            CIDEr = test_met['CIDEr']
            ROUGE_L = test_met['ROUGE_L']
            self._save_model_g(epoch,
                               g_loss, b1, b2, b3, b4, ROUGE_L, CIDEr, METEOR)
            self._save_model(epoch,
                             d_loss)
            self._save_model_1(epoch,
                             d_loss_1)
        # 迭代了200次，所以x的取值范围为(0，200)，然后再将每次相对应的准确率以及损失率附在x上

    def _init_sentence_model(self):
        model = SentenceLSTM(embed_size=self.args.embed_size,
                             hidden_size=self.args.hidden_size)

        try:
            model_state = torch.load(self.args.load_sentence_model_path)
            model.load_state_dict(model_state['sentence_model'])
            print("[Load Sentence Model From {} Succeed!\n".format(self.args.load_sentence_model_path))
        except Exception as err:
            print("[Load Sentence model Failed {}!]\n".format(err))

        if not self.args.sentence_trained:
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

    def _init_word_model(self):
        model = WordLSTM(embed_size=self.args.embed_size,
                         hidden_size=self.args.hidden_size,
                         vocab_size=len(self.vocab),
                         n_max=self.args.n_max)

        try:
            model_state = torch.load(self.args.load_word_model_path)
            model.load_state_dict(model_state['word_model'])
            print("[Load Word Model From {} Succeed!\n".format(self.args.load_word_model_path))
        except Exception as err:
            print("[Load Word model Failed {}!]\n".format(err))

        if not self.args.word_trained:
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


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    # RESUME_MODEL_PATH = '../cm/Medical_Report_Generation_with_CNN_HLSTM/models/2023-10-30 21:31/train_best.pth.tar'
    """
    Data Argument
    """
    parser.add_argument('--data_dir', type=str, default='/home/upc/sxx/Data/cov_ctr/annotation.json',
                        help='path for images')
    parser.add_argument('--patience', type=int, default=50)
    # parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--batch_size', type=int, default=16)

    # Disc Path Argument

    parser.add_argument('--disc_train_true_data_list', type=str, default='./data/new_data/disc_train_true_data.txt',
                        help='the path for True data')
    parser.add_argument('--disc_train_fake_data_list', type=str, default='./data/new_data/disc_train_fake_data.txt',
                        help='the path for Fake data')
    parser.add_argument('--adver_file_list', type=str, default='./data/new_data/val_data.txt',
                        help='the val array')
    parser.add_argument('--file_list', type=str, default='./data/new_data/adver_list.txt',
                        help='the path for test file list')
    parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.')

    # transforms argument
    parser.add_argument('--resize', type=int, default=256,
                        help='size for resizing images')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='size for randomly cropping images')

    # Disc Load/Save model argument

    parser.add_argument('--discs_model_path', type=str, default='./report_discs_models/',
                        help='path for saving disc model')
    parser.add_argument('--disc_trained', action='store_true', default=True,
                        help='Whether train disc or not')
    parser.add_argument('--load_disc_model_path', type=str, default='./report_disc_models/v4_cov/disc_train_best_loss.pth.tar',
                        help='The path of loaded disc model')
    parser.add_argument('--load_discs_model_path', type=str,
                        default='./report_discs_models/v4_cov/discs_train_best_loss.pth.tar',
                        help='The path of loaded discs model')
    parser.add_argument('--disc_saved_model_name', type=str, default='./report_disc_models/v4_cov/',
                        help='The name of saved model')
    parser.add_argument('--discs_saved_model_name', type=str, default='./report_discs_models/v4_cov/',
                        help='The name of saved model')



    # Path Argument
    parser.add_argument('--vocab_path', type=str, default='./data/new_data/vocab_cov.pkl',
                        help='the path for vocabulary object')
    parser.add_argument('--image_dir', type=str, default='./data/images',
                        help='the path for images')
    parser.add_argument('--caption_json', type=str, default='./data/new_data/captions.json',
                        help='path for captions')
    parser.add_argument('--train_file_list', type=str, default='./data/new_data/test_data.txt',
                        help='the train array')
    parser.add_argument('--val_file_list', type=str, default='./data/new_data/val_data.txt',
                        help='the val array')
    # Load/Save model argument
    parser.add_argument('--model_path', type=str, default='./report_v4_models/',
                        help='path for saving trained models')
    parser.add_argument('--load_model_path', type=str,
                        default='./report_v4_models/v4_cov/train_best_loss.pth.tar',
                        help='The path of loaded model')
    parser.add_argument('--saved_model_name', type=str, default='./report_v4_models/v4_cov/',
                        help='The name of saved model')

    # VisualFeatureExtractor
    parser.add_argument('--visual_model_name', type=str, default='resnet152',
                        help='CNN model name')
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='not using pretrained model when training')

    parser.add_argument('--load_visual_model_path', type=str,
                        default='./report_v4_models/v4_cov/train_best_loss.pth.tar')
    parser.add_argument('--visual_trained', action='store_true', default=True,
                        help='Whether train visual extractor or not')
    parser.add_argument('--num_workers', type=int, default=0, help='the number of workers for dataloader.')
    # parser.add_argument('--ann_path', type=str, default='./data/mimic_cxr/annotation.json', help='the path to the directory containing the data.')
    parser.add_argument('--max_seq_length', type=int, default=60, help='the maximum sequence length of the reports.')

    # MLC
    parser.add_argument('--classes', type=int, default=14)
    parser.add_argument('--sementic_features_dim', type=int, default=512)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--load_mlc_model_path', type=str,
                        default='./report_v4_models/v4_cov/train_best_loss.pth.tar')
    parser.add_argument('--mlc_trained', action='store_true', default=True)

    # Co-Attention
    parser.add_argument('--attention_version', type=str, default='v4')
    parser.add_argument('--load_co_model_path', type=str,
                        default='./report_v4_models/v4_cov/train_best_loss.pth.tar')
    parser.add_argument('--co_trained', action='store_true', default=True)
    # Sentence Model
    parser.add_argument('--momentum', type=int, default=0.1)
    parser.add_argument('--embed_size', type=int, default=512)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--sent_version', type=str, default='v1')
    parser.add_argument('--sentence_num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--load_sentence_model_path', type=str,
                        default='./report_v4_models/v4_cov/train_best_loss.pth.tar')
    parser.add_argument('--sentence_trained', action='store_true', default=True)

    # Word Model
    parser.add_argument('--word_num_layers', type=int, default=1)
    parser.add_argument('--load_word_model_path', type=str,
                        default='./report_v4_models/v4_cov/train_best_loss.pth.tar')
    parser.add_argument('--word_trained', action='store_true', default=True)

    # Saved result
    parser.add_argument('--result_path', type=str, default='./results_test',
                        help='the path for storing results')
    parser.add_argument('--result_path_test', type=str, default='./results_test',
                        help='the path for storing results')
    parser.add_argument('--result_name', type=str, default='generate',
                        help='the name of results')

    """
    Training Argument
    """
    # parser.add_argument('--learning_rate_gen', type=int, default=5e-5)
    parser.add_argument('--learning_rate', type=int, default=1e-5)
    parser.add_argument('--epochs', type=int, default=50)  # 1000

    parser.add_argument('--clip', type=float, default=0.35,
                        help='gradient clip, -1 means no clip (default: 0.35)')
    parser.add_argument('--s_max', type=int, default=6)
    parser.add_argument('--n_max', type=int, default=15)

    # Loss Function
    parser.add_argument('--lambda_tag', type=float, default=10000)
    parser.add_argument('--lambda_stop', type=float, default=10)
    parser.add_argument('--lambda_sentence', type=float, default=1)
    parser.add_argument('--lambda_word', type=float, default=1)
    parser.add_argument('--load_semantic_model_path', type=str, default="./report_v4_models/v4_cov/train_best_loss.pth.tar")
    parser.add_argument('--semantic_trained', action='store_true', default=False)


    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    # tokenizer = Tokenizer(args)
    adversarial = Adversarial(args)
    adversarial.train()