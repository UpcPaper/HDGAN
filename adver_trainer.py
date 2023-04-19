# -*- coding: utf-8 -*-
import time
from tqdm import tqdm
import argparse
import pickle
import torch
import torch.optim as optim
import numpy as np
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils.discriminator import *

from utils.models import *
from utils.dataset import *
from utils.loss import *
from utils.logger import Logger


class DebuggerBase:
    def __init__(self, args):
        self.args = args
        self.min_val_loss = 10000000000
        self.min_val_tag_loss = 1000000
        self.min_val_stop_loss = 1000000
        self.min_val_word_loss = 10000000

        self.min_train_loss = 10000000000
        self.min_train_tag_loss = 1000000
        self.min_train_stop_loss = 1000000
        self.min_train_word_loss = 10000000

        self.params = None
        self.params_d = None

        self._init_model_path()
        self.model_dir = self._init_model_dir()
        self.writer = self._init_writer()
        self.train_transform = self._init_train_transform()
        self.val_transform = self._init_val_transform()
        self.vocab = self._init_vocab()
        self.model_state_dict = self._load_model_state_dict()
        self.transform = self.__init_transform()

        self.data_loader = self.__init_data_loader(self.args.train_file_list) # 随机生成 the path for test file list

        # self.train_data_loader = self._init_data_loader(self.args.train_file_list, self.train_transform)
        # self.val_data_loader = self._init_data_loader(self.args.val_file_list, self.val_transform)

        self.extractor = self._init_visual_extractor()
        self.mlc = self._init_mlc()
        self.co_attention = self._init_co_attention()
        self.sentence_model = self._init_sentence_model()
        self.word_model = self._init_word_model()
        self.disc_model = self._init_disc_model()

        self.ce_criterion = self._init_ce_criterion()
        self.mse_criterion = self._init_mse_criterion()
        self.adver_loss = self._init_adver_loss()
        self.reward = torch.zeros(self.args.batch_size, 1)
        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler()  # 自动调整学习率
        self.logger = self._init_logger()
        self.writer.write("{}\n".format(self.args))

    def train(self):
        train_loss = self._epoch_train()
        self.scheduler.step(train_loss)  # 对lr进行调整
        print(
            "{} - [Update generator model] train loss:{}  - lr:{}\n".format(self._get_now(),
                                                                                train_loss,
                                                                                self.optimizer.param_groups[0]['lr']))
        # self._save_model(epoch_id,
        #                 train_loss)
        return train_loss
    def _epoch_train(self):
        raise NotImplementedError

    def _epoch_val(self):
        raise NotImplementedError

    def __init_transform(self):
        transform = transforms.Compose([
            transforms.Resize(self.args.resize),
            transforms.RandomCrop(self.args.crop_size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        return transform

    def _init_train_transform(self):
        transform = transforms.Compose([
            transforms.Resize(self.args.resize),
            transforms.RandomCrop(self.args.crop_size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        return transform

    def _init_val_transform(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.crop_size, self.args.crop_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        return transform

    def _init_model_dir(self):
        model_dir = os.path.join(self.args.model_path, self.args.saved_model_name)

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_dir = os.path.join(model_dir)

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        return model_dir

    def _init_vocab(self):
        with open(self.args.vocab_path, 'rb') as f:
            vocab = pickle.load(f)

        print("Vocabulary Size:{}\n".format(len(vocab)))

        return vocab

    def _load_model_state_dict(self):
        self.start_epoch = 0
        try:
            model_state = torch.load(self.args.load_model_path)
            self.start_epoch = model_state['epoch']
            print("[Load Model-{} Succeed!]\n".format(self.args.load_model_path))
            print("Load From Epoch {}\n".format(model_state['epoch']))
            return model_state
        except Exception as err:
            print("[Load Model Failed] {}\n".format(err))
            return None

    def _init_visual_extractor(self):
        model = VisualFeatureExtractor(model_name=self.args.visual_model_name,
                                       pretrained=self.args.pretrained)

        try:
            model_state = torch.load(self.args.load_visual_model_path)
            model.load_state_dict(model_state['extractor'])
            # print("[Load Visual Extractor Succeed!]\n")
        except Exception as err:
            print("[Load Visual Extractor Model Failed] {}\n".format(err))

        if not self.args.visual_trained:
            for i, param in enumerate(model.parameters()):
                param.requires_grad = True
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
                param.requires_grad = True
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

    def _init_disc_model(self):  # 加载判别器
        model = Discriminator(seq_length=1,
                              vocab_size=2195,
                              emb_size=32,
                              filter_size=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                              num_filter=[100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160],
                              dropoutRate=0.1)
        try:
            model_state = torch.load(self.args.load_disc_model_path)
            model.load_state_dict(model_state['discriminator'])
            # print("[Load Discriminator Succeed!]\n")
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
    def _init_discs_model(self):
        model = Discriminator(vocab_size=self.vocab_count,
                              input_size=50,
                              hidden_size=100,
                              num_class=2,
                              num_layers=1)
        try:
            model = torch.load(self.args.load_discs_model_path)
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

    def __init_data_loader(self, file_list):
        data_loader = get_loader(image_dir=self.args.image_dir,
                                 caption_json=self.args.caption_json,
                                 file_list=file_list,
                                 vocabulary=self.vocab,
                                 transform=self.transform,
                                 batch_size=self.args.batch_size,
                                 s_max=self.args.s_max,
                                 n_max=self.args.n_max,
                                 shuffle=False)
        return data_loader

    @staticmethod
    def _init_ce_criterion():
        return nn.CrossEntropyLoss(size_average=False, reduce=False)

    @staticmethod
    def _init_mse_criterion():
        return nn.MSELoss()

    @staticmethod
    def _init_adver_loss():
        return 0

    def _init_optimizer(self):
        return torch.optim.Adam(params=self.params, lr=self.args.learning_rate)

    def _log(self,
             train_tags_loss,
             train_stop_loss,
             train_word_loss,
             train_loss,
             lr,
             epoch):
        info = {
            'train tags loss': train_tags_loss,
            'train stop loss': train_stop_loss,
            'train word loss': train_word_loss,
            'train loss': train_loss,
            'learning rate': lr
        }

        for tag, value in info.items():
            self.logger.scalar_summary(tag, value, epoch + 1)

    def _init_logger(self):
        logger = Logger(os.path.join(self.model_dir, 'logs'))
        return logger

    def _init_writer(self):
        writer = open(os.path.join(self.model_dir, 'logs.txt'), 'w')
        return writer

    def _to_var(self, x, requires_grad=False):
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
        if not os.path.exists(self.args.model_path):
            os.makedirs(self.args.model_path)

    def _init_log_path(self):
        if not os.path.exists(self.args.log_path):
            os.makedirs(self.args.log_path)

    def _save_model(self,
                    epoch_id,
                    train_loss):
        def save_whole_model(_filename):
            print("Saved Model in {}\n".format(_filename))
            torch.save({'extractor': self.extractor.state_dict(),
                        'mlc': self.mlc.state_dict(),
                        'co_attention': self.co_attention.state_dict(),
                        'sentence_model': self.sentence_model.state_dict(),
                        'word_model': self.word_model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'epoch': epoch_id},
                       os.path.join(self.model_dir, "{}".format(_filename)))

        # file_name = "train_best_loss.pth.tar"
        # save_whole_model(file_name)
        # if loss < self.min_train_loss:
        #     file_name = "disc_train_best_loss.pth.tar"
        #     save_whole_model(file_name)
        #     self.min_train_loss = loss


class LSTMDebugger(DebuggerBase):
    def _init_(self, args):
        DebuggerBase.__init__(self, args)
        self.args = args
        self.reward = torch.zeros(self.args.batch_size, 1)
        self.emb_size = 32

    def __vec2sent(self, array):  # array是word_id 将Word_id转成单词
        sampled_caption = []
        for word_id in array:
            word = self.vocab.get_word_by_id(word_id)
            if word == '<start>':
                continue
            if word == '<end>' or word == '<pad>':
                break
            sampled_caption.append(word)
        return sampled_caption


    def loss_with_reward(self, prediction, x, rewards):
        embedding = nn.Embedding(2195, 2195)
        prediction = embedding(prediction.long())
        # print ("prediction after embedding:", prediction.shape)
        x1 = x.contiguous().view([-1, 1]).long()
        # print ("x1:", x1.shape)

        one_hot = torch.Tensor(x1.shape[0], 2195).cuda()
        one_hot.zero_()
        x2 = one_hot.scatter_(1, x1, 1)

        pred1 = prediction.view([-1, 2195])
        pred2 = torch.log(torch.clamp(pred1, min=1e-20, max=1.0))
        prod = torch.mul(x2.cuda(), pred2.cuda())
        reduced_prod = torch.sum(prod, dim=1)
        rewards_prod = torch.mul(reduced_prod.cuda(), rewards.view([-1]).cuda())
        generator_loss = torch.sum(rewards_prod)
        return -generator_loss
    def loss_with_reward_1(self, prediction, x, rewards):
        embedding = nn.Embedding(2195, 2195)
        prediction = embedding(prediction.long())
        # print ("prediction after embedding:", prediction.shape)
        x1 = x.contiguous().view([-1, 1]).long()
        # print ("x1:", x1.shape)

        one_hot = torch.Tensor(x1.shape[0], 2195).cuda()
        one_hot.zero_()
        x2 = one_hot.scatter_(1, x1, 1)

        pred1 = prediction.view([-1, 2195])
        pred2 = torch.log(torch.clamp(pred1, min=1e-20, max=1.0))
        prod = torch.mul(x2.cuda(), pred2.cuda())
        reduced_prod = torch.sum(prod, dim=1)
        rewards_prod = torch.mul(reduced_prod.cuda(), rewards.view([-1]).cuda())
        generator_loss = torch.sum(rewards_prod)
        return -generator_loss
    def _epoch_train(self):
        tag_loss, stop_loss, word_loss, loss = 0, 0, 0, 0
        self.extractor.train()
        self.mlc.train()
        self.co_attention.train()
        self.sentence_model.train()
        self.word_model.train()
        progress_bar = tqdm(self.data_loader, desc='Adversarial')
        for images, image_id, label, captions, prob in progress_bar:
            batch_tag_loss, batch_stop_loss, batch_word_loss, batch_sentence_loss, batch_loss = 0, 0, 0, 0, 0
            images = self._to_var(images, requires_grad=False)
            visual_features, avg_features = self.extractor.forward(images)
            tags, semantic_features = self.mlc.forward(avg_features)
            # 标签损失
            batch_tag_loss = self.mse_criterion(tags, self._to_var(label, requires_grad=False)).sum()
            sentence_states = None
            # 中间层
            prev_hidden_states = self._to_var(torch.zeros(images.shape[0], 1, self.args.hidden_size), requires_grad=False)
            prob_real = self._to_var(torch.Tensor(prob).long(), requires_grad=False)
            context = self._to_var(torch.Tensor(captions).long(), requires_grad=False)
            pred_sentences = {}  # 预测
            for i in image_id:
                pred_sentences[i] = {}  # 具体到每一张
            for sentence_index in range(0, captions.shape[1]):  # 是s_max=6 一个caption里有六句话
                ctx, alpha_v, alpha_a = self.co_attention.forward(avg_features,
                                                                  semantic_features,
                                                                  prev_hidden_states)
                topic, p_stop, hidden_states, sentence_states = self.sentence_model.forward(ctx,
                                                                                            prev_hidden_states,
                                                                                            sentence_states)
                batch_stop_loss += self.ce_criterion(p_stop.squeeze(), prob_real[:, sentence_index]).sum()

                start_tokens = np.zeros((topic.shape[0], 1))  # [4, 1]
                start_tokens[:, 0] = self.vocab('<start>')
                start_tokens = self._to_var(torch.Tensor(start_tokens).long(), requires_grad=False)
                sample_ids = self.word_model.sample(topic, start_tokens)
                reward = []
                sample_ids = torch.from_numpy(np.array(sample_ids))
                for j in range(0, sample_ids.shape[1]):
                    output = self.disc_model.forward(sample_ids[:, j:j + 1].cuda().long())
                    output = self._to_var(output, requires_grad=False)
                    indices = torch.LongTensor([0])
                    out = torch.index_select(output, 1, indices.cuda())
                    for i in out:
                        reward.append(i.item())
                reward = np.transpose(np.array(reward)) / 1.0
                reward1 = torch.Tensor(reward)
                s = []
                a = [0]
                for i in range(0, context[:, sentence_index, :].shape[0]):
                    t = context[:, sentence_index, :][i].tolist()  # 将tensor转为list
                    for j in range(0, self.args.n_max - context[:, sentence_index, :].shape[1]):
                        t.extend(a)
                    s.append(t)
                context1 = torch.Tensor(s)
                batch_sentence_loss += (self.loss_with_reward(sample_ids, context1.cuda(), reward1)).sum()
            batch_loss = self.args.lambda_tag * batch_tag_loss \
                         + self.args.lambda_stop * batch_stop_loss \
                         + self.args.lambda_word * batch_sentence_loss
            self.optimizer.zero_grad()  # 把梯度置零，也就是把loss关于weight的导数变成0
            batch_loss.backward()  # 反向传播求梯度retain_graph=True
            if self.args.clip > 0:
                # 最简单粗暴的方法，设定阈值，当梯度小于/大于阈值时，更新的梯度为阈值  梯度裁剪
                torch.nn.utils.clip_grad_norm(self.sentence_model.parameters(), self.args.clip)
                torch.nn.utils.clip_grad_norm(self.word_model.parameters(), self.args.clip)
            self.optimizer.step()  # 更新所有参数
            tag_loss += self.args.lambda_tag * batch_tag_loss
            stop_loss += self.args.lambda_stop * batch_stop_loss
            word_loss += self.args.lambda_word * batch_word_loss
            loss += batch_loss  # 根本原因
        return loss

    def _init_sentence_model(self):
        model = SentenceLSTM(version=self.args.sent_version,
                             embed_size=self.args.embed_size,
                             hidden_size=self.args.hidden_size,
                             num_layers=self.args.sentence_num_layers,
                             dropout=self.args.dropout,
                             momentum=self.args.momentum)

        try:
            model_state = torch.load(self.args.load_sentence_model_path)
            model.load_state_dict(model_state['sentence_model'])
            # print("[Load Sentence Model Succeed!\n")
        except Exception as err:
            print("[Load Sentence model Failed {}!]\n".format(err))

        if not self.args.sentence_trained:
            for i, param in enumerate(model.parameters()):
                param.requires_grad = True  # 反向传播后调节参数时(不)调节它
        else:
            if self.params:
                self.params += list(model.parameters())
            else:
                self.params = list(model.parameters())

        if self.args.cuda:
            model = model.cuda()
        return model

    def _init_word_model(self):
        model = WordLSTM(vocab_size=len(self.vocab),
                         embed_size=self.args.embed_size,
                         hidden_size=self.args.hidden_size,
                         num_layers=self.args.word_num_layers,
                         n_max=self.args.n_max)

        try:
            model_state = torch.load(self.args.load_word_model_path)
            model.load_state_dict(model_state['word_model'])
            # print("[Load Word Model Succeed!\n")
        except Exception as err:
            print("[Load Word model Failed {}!]\n".format(err))

        if not self.args.word_trained:
            for i, param in enumerate(model.parameters()):
                param.requires_grad = True
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

    """
    Data Argument
    """
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--mode', type=str, default='train')

    # Path Argument
    parser.add_argument('--vocab_path', type=str, default='./data/new_data/vocab.pkl',
                        help='the path for vocabulary object')
    parser.add_argument('--image_dir', type=str, default='./data/images',
                        help='the path for images')
    parser.add_argument('--caption_json', type=str, default='./data/new_data/captions.json',
                        help='path for captions')
    parser.add_argument('--train_file_list', type=str, default='./data/new_data/adver_list.txt',
                        help='the train array')
    parser.add_argument('--val_file_list', type=str, default='./data/new_data/val_data.txt',
                        help='the val array')
    # transforms argument
    parser.add_argument('--resize', type=int, default=256,
                        help='size for resizing images')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='size for randomly cropping images')

    # Disc Load/Save model argument

    parser.add_argument('--disc_model_path', type=str, default='./report_disc_models/',
                        help='path for saving disc trained models')
    parser.add_argument('--discs_model_path', type=str, default='./report_discs_models/',
                        help='path for saving disc model')
    parser.add_argument('--load_discs_model_path', type=str, default='.',
                        help='The path of loaded discs model')
    parser.add_argument('--disc_trained', action='store_true', default=True,
                        help='Whether train disc or not')
    parser.add_argument('--load_disc_model_path', type=str,
                        default='./report_disc_models/v4/disc_train_best_loss.pth.tar',
                        help='The path of loaded model')
    parser.add_argument('--disc_saved_model_name', type=str, default='./report_disc_models/v4/',
                        help='The name of saved model')

    # Load/Save model argument
    parser.add_argument('--model_path', type=str, default='./report_v4_models/v4/',
                        help='path for saving trained models')
    parser.add_argument('--load_model_path', type=str, default='v4/train_best_loss.pth.tar',
                        help='The path of loaded model')
    parser.add_argument('--saved_model_name', type=str, default='v4',
                        help='The name of saved model')

    """
    Model Argument
    """
    parser.add_argument('--momentum', type=int, default=0.1)
    # VisualFeatureExtractor
    parser.add_argument('--visual_model_name', type=str, default='resnet152',
                        help='CNN model name')
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='not using pretrained model when training')
    parser.add_argument('--load_visual_model_path', type=str,
                        default='.')
    parser.add_argument('--visual_trained', action='store_true', default=False,
                        help='Whether train visual extractor or not')

    # MLC
    parser.add_argument('--classes', type=int, default=210)
    parser.add_argument('--sementic_features_dim', type=int, default=512)
    parser.add_argument('--k', type=int, default=6)
    parser.add_argument('--load_mlc_model_path', type=str,
                        default='.')
    parser.add_argument('--mlc_trained', action='store_true', default=False)

    # Co-Attention
    parser.add_argument('--attention_version', type=str, default='v4')
    parser.add_argument('--embed_size', type=int, default=512)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--load_co_model_path', type=str, default='.')
    parser.add_argument('--co_trained', action='store_true', default=False)

    # Sentence Model
    parser.add_argument('--sent_version', type=str, default='v1')
    parser.add_argument('--sentence_num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--load_sentence_model_path', type=str,
                        default='.')
    parser.add_argument('--sentence_trained', action='store_true', default=False)

    # Word Model
    parser.add_argument('--word_num_layers', type=int, default=2)
    parser.add_argument('--load_word_model_path', type=str,
                        default='.')
    parser.add_argument('--word_trained', action='store_true', default=False)

    """
    Training Argument
    """
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=int, default=0.001)
    parser.add_argument('--epochs', type=int, default=1)  # 1000

    parser.add_argument('--clip', type=float, default=-1,
                        help='gradient clip, -1 means no clip (default: 0.35)')
    parser.add_argument('--s_max', type=int, default=6)
    parser.add_argument('--n_max', type=int, default=50)

    # Loss Function
    parser.add_argument('--lambda_tag', type=float, default=10000)
    parser.add_argument('--lambda_stop', type=float, default=10)
    parser.add_argument('--lambda_word', type=float, default=1)

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    debugger = LSTMDebugger(args)
    debugger.train()
