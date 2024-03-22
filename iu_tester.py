# -*- coding: utf-8 -*-
import sys
import time
import os
import pickle
import random
import argparse
from tqdm import tqdm
from PIL import Image
# import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from metric_performance import compute_scores

from utils.models import *
from utils.dataset_cov import *
from utils.loss import *
from utils.build_tag import *


class CaptionSampler(object):
    def __init__(self, args):
        self.args = args

        self.vocab = self.__init_vocab()
        self.tagger = self.__init_tagger()
        self.transform = self.__init_transform()
        self.data_loader = self.__init_data_loader(split='test', transform=self.transform, shuffle=True) # 随机生成 the path for test file list
        self.model_state_dict = self.__load_mode_state_dict()

        self.extractor = self.__init_visual_extractor()
        self.semantic = self.__init_semantic_embedding()
        # self.mlc = self.__init_mlc()
        # self.co_attention = self.__init_co_attention()
        self.sentence_model = self.__init_sentence_model()
        self.word_model = self.__init_word_model()
        self.ce_criterion = self._init_ce_criterion()
        self.mse_criterion = self._init_mse_criterion()
        self.writer = self._init_writer()
        self.writer_true = self._init_writer_true()
    @staticmethod
    def _init_ce_criterion():
        return nn.CrossEntropyLoss(size_average=False, reduce=False)

    @staticmethod
    def _init_mse_criterion():
        return nn.MSELoss()

    # def rand_inputofG(self, file):  # 随机写入500
    #     with open('./data/new_data/image_name.txt', 'r') as f:
    #         lines = f.readlines()
    #
    #     fa = open(file, 'w')
    #     for _ in range(500):
    #         fa.write(lines.pop(random.randint(0, len(lines) - 1)))
    #     return fa

    def _init_writer(self):
        writer = open('./data/new_data/disc_train_fake_data.txt', 'w')
        return writer
    def _init_writer_true(self):
        writer = open('./data/new_data/disc_train_true_data.txt', 'w')
        return writer

    def test(self):
        tag_loss, stop_loss, word_loss, loss = 0, 0, 0, 0
        # self.extractor.eval()
        # self.mlc.eval()
        # self.co_attention.eval()
        # self.sentence_model.eval()
        # self.word_model.eval()

        for i, (images, _, label, captions, prob) in enumerate(self.data_loader):
            batch_tag_loss, batch_stop_loss, batch_word_loss, batch_loss = 0, 0, 0, 0
            images = self.__to_var(images, requires_grad=False)

            # visual_features, avg_features = self.extractor.forward(images)
            # tags, semantic_features = self.mlc.forward(avg_features)
            frontal, lateral, avg = self.extractor.forward(images, images)
            state_c, state_h = self.semantic.forward(avg)
            state = (torch.unsqueeze(state_c, 0), torch.unsqueeze(state_h, 0))
            pre_hid = torch.unsqueeze(state_h, 1)
            # batch_tag_loss = self.mse_criterion(tags, self.__to_var(label, requires_grad=False)).sum()

            sentence_states = None
            prev_hidden_states = self.__to_var(torch.zeros(images.shape[0], 1, self.args.hidden_size))

            captions = self.__to_var(torch.Tensor(captions).long(), requires_grad=False)
            prob_real = self.__to_var(torch.Tensor(prob).long(), requires_grad=False)

            for sentence_index in range(captions.shape[1]):
                # ctx, v_att, a_att = self.co_attention.forward(avg_features,
                #                                        semantic_features,
                #                                        prev_hidden_states)

                _,p_stop, state, h0_word, c0_word, pre_hid = self.sentence_model.forward(frontal, lateral, state, pre_hid)

                batch_stop_loss += self.ce_criterion(p_stop.squeeze(), prob_real[:, sentence_index]).sum()
                state_word = (c0_word, h0_word)

                for word_index in range(captions.shape[2] - 1):
                    word, state_word = self.word_model.forward(captions[:, sentence_index, word_index], state_word)
                    word_mask = (captions[:, sentence_index, word_index + 1] > 0).float()
                    batch_word_loss += (
                                self.ce_criterion(word, captions[:, sentence_index, word_index + 1]) * word_mask).mean()

            # batch_loss = self.args.lambda_tag * batch_tag_loss \
            #              + self.args.lambda_stop * batch_stop_loss \
            #              + self.args.lambda_word * batch_word_loss
            batch_stop_loss = batch_stop_loss * 20
            batch_word_loss = batch_word_loss
            batch_loss = (batch_word_loss + batch_stop_loss) / 2

            # tag_loss += self.args.lambda_tag * batch_tag_loss.data
            stop_loss += self.args.lambda_stop * batch_stop_loss.data
            word_loss += self.args.lambda_word * batch_word_loss.data
            loss += batch_loss.data

        return tag_loss, stop_loss, word_loss, loss

    def generate(self):
        self.extractor.train()
        # self.mlc.train()
        # self.co_attention.train()
        self.semantic.train()
        self.sentence_model.train()
        self.word_model.train()

        progress_bar = tqdm(self.data_loader, desc='Generating')
        results = {}
        for images1, images2, captions, prob, image_id in progress_bar:
            images_frontal = self.__to_var(images1, requires_grad=False)
            images_lateral = self.__to_var(images2, requires_grad=False)
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
                start_tokens = self.__to_var(torch.Tensor(start_tokens).long(), requires_grad=False)
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
            #     with open("./data/new_data/all_true_data_mimic.json", 'r') as fj:
            #         data = json.load(fj)
            #         with open('./data/new_data/disc_train_true_data.txt', 'w') as ft:
            #             ft.writelines(data[i]+"\n")
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
                with open("./data/new_data/all_true_data_cov.json", 'r') as fj:
                    data = json.load(fj)
                    # print(id)
                    # print(type(str(id)))
                    # print(data[str(id)])
                    self.writer_true.write(data[str(id)])
                # print(id)
                # print("pred_sentences", pred_sentences[id])
                # print("=====================================================")
                self.writer.write(str(pred_sentences[id]) + "." + "\n")

        self.__save_json(results)
        with open("./results_test/generate.json", 'r') as f:
            results = json.load(f)
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


    # def sample(self, image_file):
    #     self.extractor.eval()
    #     self.mlc.eval()
    #     self.co_attention.eval()
    #     self.sentence_model.eval()
    #     self.word_model.eval()
    #
    #     cam_dir = self.__init_cam_path(image_file)
    #     image_file = os.path.join(self.args.image_dir, image_file)
    #
    #     imageData = Image.open(image_file).convert('RGB')
    #     imageData = self.transform(imageData)
    #     imageData = imageData.unsqueeze_(0)
    #
    #     image = self.__to_var(imageData, requires_grad=False)
    #
    #     visual_features, avg_features = self.extractor.forward(image)
    #     avg_features.unsqueeze_(0)
    #
    #     tags, semantic_features = self.mlc(avg_features)
    #     sentence_states = None
    #     prev_hidden_states = self.__to_var(torch.zeros(1, 1, self.args.hidden_size))
    #
    #     pred_sentences = []
    #
    #     for i in range(self.args.s_max):
    #         ctx, alpha_v, alpha_a = self.co_attention.forward(avg_features, semantic_features, prev_hidden_states)
    #         topic, p_stop, hidden_state, sentence_states = self.sentence_model.forward(ctx,
    #                                                                                    prev_hidden_states,
    #                                                                                    sentence_states)
    #         p_stop = p_stop.squeeze(1)
    #         p_stop = torch.max(p_stop, 1)[1].unsqueeze(1)
    #         # print(type(p_stop)) <class 'torch.autograd.variable.Variable'>
    #         start_tokens = np.zeros((topic.shape[0], 1))
    #         start_tokens[:, 0] = self.vocab('<start>')
    #         start_tokens = self.__to_var(torch.Tensor(start_tokens).long(), requires_grad=False)
    #
    #         sampled_ids = self.word_model.sample(topic, start_tokens)
    #         prev_hidden_states = hidden_state
    #         p_stop = p_stop.cpu().data.numpy()  # 将p_stop 转换为numpy数组
    #         sampled_ids = sampled_ids * p_stop
    #         # print(type(sampled_ids))  # <class 'numpy.ndarray'>
    #         # sampled_ids = Variable(sampled_ids) sampled_ids.cpu().detach().numpy()[0])
    #         # sampled_ids.astype(np.float64)
    #         # sampled_ids = Variable(torch.from_numpy(sampled_ids))
    #         sampled_ids = Variable(torch.from_numpy(sampled_ids), requires_grad=True)
    #         pred_sentences.append(self.__vec2sent(sampled_ids.cpu().data.numpy()[0]))
    #
    #         cam = torch.mul(visual_features, alpha_v.view(alpha_v.shape[0], alpha_v.shape[1], 1, 1)).sum(1)
    #         cam.squeeze_()
    #
    #         cam = cam.cpu().data.numpy()
    #         cam = cam / np.sum(cam)
    #         cam = cv2.resize(cam, (self.args.cam_size, self.args.cam_size))
    #         cam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    #
    #         imgOriginal = cv2.imread(image_file, 1)
    #         imgOriginal = cv2.resize(imgOriginal, (self.args.cam_size, self.args.cam_size))
    #
    #         img = cam * 0.5 + imgOriginal
    #         cv2.imwrite(os.path.join(cam_dir, '{}.png'.format(i)), img)
    #     print("pred sentences", pred_sentences)
    #     return '. '.join(pred_sentences)

    def _generate_cam(self, images_id, visual_features, alpha_v, sentence_id):
        alpha_v *= 100
        cam = torch.mul(visual_features, alpha_v.view(alpha_v.shape[0], alpha_v.shape[1], 1, 1)).sum(1)
        cam.squeeze_()
        cam = cam.cpu().data.numpy()
        for i in range(cam.shape[0]):
            image_id = images_id[i]
            cam_dir = self.__init_cam_path(images_id[i])

            org_img = cv2.imread(os.path.join(self.args.image_dir, image_id), 1)
            org_img = cv2.resize(org_img, (self.args.cam_size, self.args.cam_size))

            heatmap = cam[i]
            heatmap = heatmap / np.max(heatmap)
            heatmap = cv2.resize(heatmap, (self.args.cam_size, self.args.cam_size))
            heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

            img = heatmap * 0.5 + org_img
            cv2.imwrite(os.path.join(cam_dir, '{}.png'.format(sentence_id)), img)

    def __init_cam_path(self, image_file):
        generate_dir = os.path.join(self.args.model_dir, self.args.generate_dir)
        if not os.path.exists(generate_dir):
            os.makedirs(generate_dir)

        image_dir = os.path.join(generate_dir, image_file)

        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        return image_dir

    def __save_json(self, result):
        result_path = self.args.result_path
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        with open(os.path.join(result_path, '{}.json'.format(self.args.result_name)), 'w') as f:
            json.dump(result, f)  # 将json信息写进文件  dump

    def __load_mode_state_dict(self):
        try:
            model_state_dict = torch.load(os.path.join(self.args.model_dir, self.args.load_model_path))
            print("[Load Model-{} Succeed!]".format(self.args.load_model_path))  # train_best_loss.pth.tar
            print("Load From Epoch {}".format(model_state_dict['epoch']))
            return model_state_dict
        except Exception as err:
            print("[Load Model Failed] {}".format(err))
            raise err

    def __init_tagger(self):
        return Tag()

    def __vec2sent(self, array):  # array是word_id 将Word_id转成单词
        sampled_caption = []
        for word_id in array:
            word = self.vocab.get_word_by_id(word_id)
            if word == '<start>':
                continue
            if word == '<end>' or word == '<pad>':
                break
            sampled_caption.append(word)
        return ' '.join(sampled_caption)

    def __init_vocab(self):
        with open(self.args.vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        print("Vocabulary Size:{}\n".format(len(vocab)))

        return vocab

    def __init_data_loader(self, split, transform, shuffle):
        data_loader = get_loader(data_dir=self.args.data_dir,
                                 split=split,
                                 vocabulary=self.vocab,
                                 transform=transform,
                                 batch_size=self.args.batch_size,
                                 s_max=self.args.s_max,
                                 n_max=self.args.n_max,
                                 shuffle=shuffle)
        return data_loader

    def __init_transform(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.resize, self.args.resize)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        return transform

    def __to_var(self, x, requires_grad=True):
        if self.args.cuda:
            x = x.cuda()
        return Variable(x, requires_grad=requires_grad)

    def __init_visual_extractor(self):
        model = VisualFeatureExtractor(self.args.embed_size)

        if self.model_state_dict is not None:
            model.load_state_dict(self.model_state_dict['extractor'])
            print("Visual Extractor Loaded!")
        if self.args.cuda:
            model = model.cuda()

        return model

    def __init_semantic_embedding(self):
        model = SemanticEmbedding(embed_size=self.args.embed_size)
        if self.model_state_dict is not None:
            model.load_state_dict(self.model_state_dict['semantic'])
            print("Visual Extractor Loaded!")
        if self.args.cuda:
            model = model.cuda()
        return model

    def __init_mlc(self):
        model = MLC(classes=self.args.classes,
                    sementic_features_dim=self.args.sementic_features_dim,
                    fc_in_features=self.extractor.out_features,
                    k=self.args.k)

        if self.model_state_dict is not None:
            print("MLC Loaded!")
            model.load_state_dict(self.model_state_dict['mlc'])

        if self.args.cuda:
            model = model.cuda()

        return model

    def __init_co_attention(self):
        model = CoAttention(version=self.args.attention_version,
                            embed_size=self.args.embed_size,
                            hidden_size=self.args.hidden_size,
                            visual_size=self.extractor.out_features,
                            k=self.args.k,
                            momentum=self.args.momentum)

        if self.model_state_dict is not None:
            print("Co-Attention Loaded!")
            model.load_state_dict(self.model_state_dict['co_attention'])

        if self.args.cuda:
            model = model.cuda()

        return model

    def __init_sentence_model(self):
        model = SentenceLSTM(embed_size=self.args.embed_size,
                             hidden_size=self.args.hidden_size)

        if self.model_state_dict is not None:
            print("Sentence Model Loaded!")
            model.load_state_dict(self.model_state_dict['sentence_model'])

        if self.args.cuda:
            model = model.cuda()
        return model

    def __init_word_model(self):
        model = WordLSTM(embed_size=self.args.embed_size,
                         hidden_size=self.args.hidden_size,
                         vocab_size=len(self.vocab),
                         n_max=self.args.n_max)
        if self.model_state_dict is not None:
            print("Word Model Loaded!")
            model.load_state_dict(self.model_state_dict['word_model'])
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
    # Path Argument
    parser.add_argument('--data_dir', type=str, default='/home/upc/sxx/Data/cov_ctr/annotation.json',
                        help='path for images')
    parser.add_argument('--model_dir', type=str, default='./report_v4_models/v4_cov/')  #   20190829-13:39/  ./report_v4_models/v4/20190802-07:33/  ./report_v4_models/v4/20190724-02:44/
    parser.add_argument('--image_dir', type=str, default='./data/images',
                        help='the path for images')
    parser.add_argument('--caption_json', type=str, default='./data/new_data/captions.json',
                        help='path for captions')
    parser.add_argument('--vocab_path', type=str, default='./data/new_data/vocab_cov.pkl',
                        help='the path for vocabulary object')
    parser.add_argument('--file_lits', type=str, default='./data/new_data/val_data.txt',
                        help='the path for test file list')
    parser.add_argument('--load_model_path', type=str, default='train_best_loss.pth.tar',
                        help='The path of loaded model')

    # transforms argument
    parser.add_argument('--resize', type=int, default=224,
                        help='size for resizing images')

    # CAM  是什么？？？
    parser.add_argument('--cam_size', type=int, default=224)
    parser.add_argument('--generate_dir', type=str, default='cam')

    # Saved result
    parser.add_argument('--result_path', type=str, default='./results_test',
                        help='the path for storing results')
    parser.add_argument('--result_name', type=str, default='generate',
                        help='the name of results')

    """
    Model argument
    """
    parser.add_argument('--momentum', type=int, default=0.1)
    # VisualFeatureExtractor
    parser.add_argument('--visual_model_name', type=str, default='resnet152',
                        help='CNN model name')
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='not using pretrained model when training')

    # MLC
    parser.add_argument('--classes', type=int, default=210)   # 210个标签
    parser.add_argument('--sementic_features_dim', type=int, default=512)
    parser.add_argument('--k', type=int, default=10)

    # Co-Attention
    parser.add_argument('--attention_version', type=str, default='v1')
    parser.add_argument('--embed_size', type=int, default=512)
    parser.add_argument('--hidden_size', type=int, default=512)

    # Sentence Model
    parser.add_argument('--sent_version', type=str, default='v1')
    parser.add_argument('--sentence_num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)

    # Word Model
    parser.add_argument('--word_num_layers', type=int, default=1)

    """
    Generating Argument
    """
    parser.add_argument('--s_max', type=int, default=10)
    parser.add_argument('--n_max', type=int, default=15)

    parser.add_argument('--batch_size', type=int, default=16)

    # Loss function
    parser.add_argument('--lambda_tag', type=float, default=10000)
    parser.add_argument('--lambda_stop', type=float, default=10)
    parser.add_argument('--lambda_word', type=float, default=1)

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    # print(args)

    # sampler = CaptionSampler(args)
    #
    # # sampler.sample('CXR1000_IM-0003-1001.png')  # 第一幅图片
    # sampler.generate()
    # sys.exit()
    f = open('./results/results.txt', 'r')

    lines = f.readlines()  # 把每一行的内容变为集合lines的一个元素
    f.close()

    for i, line in enumerate(lines):
        if i%3 == 1:
            lines[i] = str(lines[i]).replace('{', '').replace('}', '')  # 去除[],这两行按数据不同，可以选择
            lines[i] = str(lines[i]).replace('0:', '').replace('1:', '').replace('2:', '').replace('3:', '').replace('4:', '').replace('5:', '').replace('6:', '').replace('7:', '').replace('8:', '').replace('9:', '')
            lines[i] = str(lines[i]).replace("'", '')  # 去除单引号,每行末尾追加换行符
        elif i % 3 == 2:
            lines[i] = str(lines[i]).replace('{', '').replace('}', '')  # 去除[],这两行按数据不同，可以选择
            lines[i] = str(lines[i]).replace('0:', '').replace('1:', '').replace('2:', '').replace('3:', '').replace('4:', '').replace('5:', '').replace('6:', '').replace('7:', '').replace('8:', '').replace('9:', '')
            lines[i] = str(lines[i]).replace("'", '')  # 去除单引号,每行末尾追加换行符
    f = open('./results/results.txt', 'w')
    f.writelines(lines)
    f.close()

    # 操作 disc_fake
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

    # 操作 disc_fake_D
    # with open('./data/new_data/disc_train_fake_data_D.txt', 'r') as fr:
    #     lines = fr.readlines()
    # for i, line in enumerate(lines):
    #     lines[i] = str(lines[i]).replace('{', '').replace('}', '')  # 去除[],这两行按数据不同，可以选择
    #     lines[i] = str(lines[i]).replace('0:', '').replace('1:', '').replace('2:', '').replace('3:', '').replace(
    #         '4:', '').replace('5:', '')
    #     lines[i] = str(lines[i]).replace("'", '')  # 去除单引号,每行末尾追加换行符
    # f = open('./data/new_data/disc_train_fake_data_D.txt', 'w')
    # f.writelines(lines)
    # f.close()