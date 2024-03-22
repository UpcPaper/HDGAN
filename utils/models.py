# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torchvision
import numpy as np
from torch.autograd import Variable
import torchvision.models as models
import sys


# class VisualFeatureExtractor(nn.Module):
#     def __init__(self, model_name='resnet152', pretrained=False):
#         super(VisualFeatureExtractor, self).__init__()
#         self.model_name = model_name
#         self.pretrained = pretrained
#         self.model, self.out_features, self.avg_func, self.bn, self.linear = self.__get_model()
#         self.activation = nn.ReLU()  # 激活函数
#
#     def __get_model(self):
#         model = None
#         out_features = None
#         func = None
#         if self.model_name == 'resnet152':
#             resnet = models.resnet152(pretrained=self.pretrained)  # 自动下载
#             modules = list(resnet.children())[:-2]  # 去掉resnet最后两层？
#             model = nn.Sequential(*modules)  # 重新生成一个新模型 快速搭建神经网络
#             out_features = resnet.fc.in_features
#             func = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
#         elif self.model_name == 'densenet201':
#             densenet = models.densenet201(pretrained=self.pretrained)
#             modules = list(densenet.features)
#             model = nn.Sequential(*modules)
#             func = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
#             out_features = densenet.classifier.in_features
#         linear = nn.Linear(in_features=out_features, out_features=out_features)
#         bn = nn.BatchNorm1d(num_features=out_features, momentum=0.1)
#         return model, out_features, func, bn, linear
#
#     def forward(self, images):
#         """
#         :param images:
#         :return:
#         """
#         visual_features = self.model(images)
#         avg_features = self.avg_func(visual_features).squeeze()
#         # avg_features = self.activation(self.bn(self.linear(avg_features)))
#
#         return visual_features, avg_features
class VisualFeatureExtractor(nn.Module):
    def __init__(self, embed_size):
        super(VisualFeatureExtractor, self).__init__()

        # frontal
        resnet_frontal = models.resnet50(pretrained=True)
        self.resnet_conv_frontal = nn.Sequential(*list(resnet_frontal.children())[:-2])
        self.avgpool_fun_frontal = nn.Sequential(* list(resnet_frontal.children())[-2:-1])
        self.dropout_frontal = nn.Dropout(0.2)
        self.affine_frontal_a = nn.Linear(2048, embed_size)
        self.affine_frontal_b = nn.Linear(2048, embed_size)

        # lateral
        resnet_lateral = models.resnet50(pretrained=True)
        self.resnet_conv_lateral = nn.Sequential(*list(resnet_lateral.children())[:-2])
        self.avgpool_fun_lateral = nn.Sequential(* list(resnet_lateral.children())[-2:-1])
        self.dropout_lateral = nn.Dropout(0.2)
        self.affine_lateral_a = nn.Linear(2048, embed_size)
        self.affine_lateral_b = nn.Linear(2048, embed_size)

        self.relu = nn.ReLU()
        self.affine = nn.Linear(2 * embed_size, embed_size)
        self.init_weights()

    def init_weights(self):
        """Initialize the weights."""
        self.affine_frontal_a.weight.data.uniform_(-0.1, 0.1)
        self.affine_frontal_a.bias.data.fill_(0)
        self.affine_frontal_b.weight.data.uniform_(-0.1, 0.1)
        self.affine_frontal_b.bias.data.fill_(0)
        self.affine_lateral_a.weight.data.uniform_(-0.1, 0.1)
        self.affine_lateral_a.bias.data.fill_(0)
        self.affine_lateral_b.weight.data.uniform_(-0.1, 0.1)
        self.affine_lateral_b.bias.data.fill_(0)
        self.affine.weight.data.uniform_(-0.1, 0.1)
        self.affine.bias.data.fill_(0)

    def forward(self, image_frontal, image_lateral):
        A_frontal = self.resnet_conv_frontal(image_frontal)  # [bs, 2048, 7, 7]
        V_frontal = A_frontal.view(A_frontal.size(0), A_frontal.size(1), -1).transpose(1, 2)  # [bs, 49, 2048]
        V_frontal = self.relu(self.affine_frontal_a(self.dropout_frontal(V_frontal)))  # [bs, 49, 512]
        avg_frontal = self.avgpool_fun_frontal(A_frontal).squeeze(2).squeeze(2)  # [bs, 2048]
        avg_frontal = self.relu(self.affine_frontal_b(self.dropout_frontal(avg_frontal)))  # [bs, 512]
        A_lateral = self.resnet_conv_lateral(image_lateral)
        V_lateral = A_lateral.view(A_lateral.size(0), A_lateral.size(1), -1).transpose(1, 2)
        V_lateral = self.relu(self.affine_lateral_a(self.dropout_lateral(V_lateral)))
        # print(self.avgpool_fun_lateral(A_lateral).shape)
        avg_lateral = self.avgpool_fun_lateral(A_lateral).squeeze(2).squeeze(2)
        avg_lateral = self.relu(self.affine_lateral_b(self.dropout_lateral(avg_lateral)))
        # print(str(avg_frontal.shape) + " " + str(avg_lateral.shape))
        avg = torch.cat((avg_frontal, avg_lateral), dim=1)
        avg = self.relu(self.affine(avg))
        return V_frontal, V_lateral, avg


class SemanticEmbedding(nn.Module):
    def __init__(self, embed_size):
        super(SemanticEmbedding, self).__init__()
        self.bn = nn.BatchNorm1d(num_features=embed_size, momentum=0.1)
        self.w1 = nn.Linear(in_features=embed_size, out_features=embed_size)
        self.w2 = nn.Linear(in_features=embed_size, out_features=embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.__init_weight()

    def __init_weight(self):
        self.w1.weight.data.uniform_(-0.1, 0.1)
        self.w1.bias.data.fill_(0)
        self.w2.weight.data.uniform_(-0.1, 0.1)
        self.w2.bias.data.fill_(0)

    def forward(self, avg):
        state_h = self.bn(self.w2(self.w1(avg)))
        return state_h, state_h

class MLC(nn.Module):
    def __init__(self,
                 classes=14, #类
                 sementic_features_dim=512,
                 fc_in_features=2048,
                 k=10):
        super(MLC, self).__init__()
        self.classifier = nn.Linear(in_features=fc_in_features, out_features=classes)
        self.embed = nn.Embedding(classes, sementic_features_dim)
        self.k = k
        self.softmax = nn.Softmax()
        self.__init_weight()

    def __init_weight(self):
        self.classifier.weight.data.uniform_(-0.1, 0.1)
        self.classifier.bias.data.fill_(0)

    def forward(self, avg_features):
        tags = self.softmax(self.classifier(avg_features))
        semantic_features = self.embed(torch.topk(tags, self.k)[1])  # 沿给定dim维度返回输入张量input中 k 个最大值
        return tags, semantic_features


class CoAttention(nn.Module):
    def __init__(self,
                 version='v1',
                 embed_size=512,  # 语义特征
                 hidden_size=512,  # 隐藏层
                 visual_size=2048,  # 视觉特征
                 k=10,
                 momentum=0.1):
        super(CoAttention, self).__init__()
        self.version = version
        self.W_v = nn.Linear(in_features=visual_size, out_features=visual_size)
        self.bn_v = nn.BatchNorm1d(num_features=visual_size, momentum=momentum)

        self.W_v_h = nn.Linear(in_features=hidden_size, out_features=visual_size)
        self.bn_v_h = nn.BatchNorm1d(num_features=visual_size, momentum=momentum)

        self.W_v_att = nn.Linear(in_features=visual_size, out_features=visual_size)
        self.bn_v_att = nn.BatchNorm1d(num_features=visual_size, momentum=momentum)

        self.W_a = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.bn_a = nn.BatchNorm1d(num_features=k, momentum=momentum)

        self.W_a_h = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.bn_a_h = nn.BatchNorm1d(num_features=1, momentum=momentum)

        self.W_a_att = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.bn_a_att = nn.BatchNorm1d(num_features=k, momentum=momentum)

        # self.W_fc = nn.Linear(in_features=visual_size, out_features=embed_size)  # for v3
        self.W_fc = nn.Linear(in_features=visual_size + hidden_size, out_features=embed_size)
        self.bn_fc = nn.BatchNorm1d(num_features=embed_size, momentum=momentum)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

        self.__init_weight()

    def __init_weight(self):
        self.W_v.weight.data.uniform_(-0.1, 0.1)
        self.W_v.bias.data.fill_(0)

        self.W_v_h.weight.data.uniform_(-0.1, 0.1)
        self.W_v_h.bias.data.fill_(0)

        self.W_v_att.weight.data.uniform_(-0.1, 0.1)
        self.W_v_att.bias.data.fill_(0)

        self.W_a.weight.data.uniform_(-0.1, 0.1)
        self.W_a.bias.data.fill_(0)

        self.W_a_h.weight.data.uniform_(-0.1, 0.1)
        self.W_a_h.bias.data.fill_(0)

        self.W_a_att.weight.data.uniform_(-0.1, 0.1)
        self.W_a_att.bias.data.fill_(0)

        self.W_fc.weight.data.uniform_(-0.1, 0.1)
        self.W_fc.bias.data.fill_(0)

    def forward(self, avg_features, semantic_features, h_sent):
        if self.version == 'v1':
            return self.v1(avg_features, semantic_features, h_sent)
        elif self.version == 'v2':
            return self.v2(avg_features, semantic_features, h_sent)
        elif self.version == 'v3':
            return self.v3(avg_features, semantic_features, h_sent)
        elif self.version == 'v4':
            return self.v4(avg_features, semantic_features, h_sent)
        elif self.version == 'v5':
            return self.v5(avg_features, semantic_features, h_sent)

    def v1(self, avg_features, semantic_features, h_sent):
        """
        only training
        :rtype: object
        """
        W_v = self.bn_v(self.W_v(avg_features))
        W_v_h = self.bn_v_h(self.W_v_h(h_sent.squeeze(1)))

        alpha_v = self.softmax(self.bn_v_att(self.W_v_att(self.tanh(W_v + W_v_h))))
        v_att = torch.mul(alpha_v, avg_features)

        W_a_h = self.bn_a_h(self.W_a_h(h_sent))
        W_a = self.bn_a(self.W_a(semantic_features))
        alpha_a = self.softmax(self.bn_a_att(self.W_a_att(self.tanh(torch.add(W_a_h, W_a)))))
        a_att = torch.mul(alpha_a, semantic_features).sum(1)

        ctx = self.W_fc(torch.cat([v_att, a_att], dim=1))

        return ctx, alpha_v, alpha_a

    def v2(self, avg_features, semantic_features, h_sent) :
        """
        no bn
        :rtype: object
        """
        W_v = self.W_v(avg_features)
        W_v_h = self.W_v_h(h_sent.squeeze(1))

        alpha_v = self.softmax(self.W_v_att(self.tanh(W_v + W_v_h)))
        v_att = torch.mul(alpha_v, avg_features)

        W_a_h = self.W_a_h(h_sent)
        W_a = self.W_a(semantic_features)
        alpha_a = self.softmax(self.W_a_att(self.tanh(torch.add(W_a_h, W_a))))
        a_att = torch.mul(alpha_a, semantic_features).sum(1)

        ctx = self.W_fc(torch.cat([v_att, a_att], dim=1))

        return ctx, alpha_v, alpha_a

    def v3(self, avg_features, semantic_features, h_sent) :
        """

        :rtype: object
        """
        W_v = self.bn_v(self.W_v(avg_features))
        W_v_h = self.bn_v_h(self.W_v_h(h_sent.squeeze(1)))

        alpha_v = self.softmax(self.W_v_att(self.tanh(W_v + W_v_h)))
        v_att = torch.mul(alpha_v, avg_features)

        W_a_h = self.bn_a_h(self.W_a_h(h_sent))
        W_a = self.bn_a(self.W_a(semantic_features))
        alpha_a = self.softmax(self.W_a_att(self.tanh(torch.add(W_a_h, W_a))))
        a_att = torch.mul(alpha_a, semantic_features).sum(1)

        ctx = self.W_fc(torch.cat([v_att, a_att], dim=1))

        return ctx, alpha_v, alpha_a

    def v4(self, avg_features, semantic_features, h_sent):
        W_v = self.W_v(avg_features)
        W_v_h = self.W_v_h(h_sent.squeeze(1))

        alpha_v = self.softmax(self.W_v_att(self.tanh(torch.add(W_v, W_v_h))))
        v_att = torch.mul(alpha_v, avg_features)

        W_a_h = self.W_a_h(h_sent)
        W_a = self.W_a(semantic_features)
        alpha_a = self.softmax(self.W_a_att(self.tanh(torch.add(W_a_h, W_a))))
        a_att = torch.mul(alpha_a, semantic_features).sum(1)

        ctx = self.W_fc(torch.cat([v_att, a_att], dim=1))

        return ctx, alpha_v, alpha_a

    def v5(self, avg_features, semantic_features, h_sent):
        W_v = self.W_v(avg_features)
        W_v_h = self.W_v_h(h_sent.squeeze(1))

        alpha_v = self.softmax(self.W_v_att(self.tanh(self.bn_v(torch.add(W_v, W_v_h)))))
        v_att = torch.mul(alpha_v, avg_features)

        W_a_h = self.W_a_h(h_sent)
        W_a = self.W_a(semantic_features)
        alpha_a = self.softmax(self.W_a_att(self.tanh(self.bn_a(torch.add(W_a_h, W_a)))))
        a_att = torch.mul(alpha_a, semantic_features).sum(1)

        ctx = self.W_fc(torch.cat([v_att, a_att], dim=1))

        return ctx, alpha_v, alpha_a


class VisualAttention(nn.Module):
    def __init__(self,
                 version='v1',
                 embed_size=512,  # 语义特征
                 hidden_size=512,  # 隐藏层
                 visual_size=2048,  # 视觉特征
                 k=10,
                 momentum=0.1):
        super(VisualAttention, self).__init__()
        self.version = version
        self.W_v = nn.Linear(in_features=visual_size, out_features=visual_size)
        self.bn_v = nn.BatchNorm1d(num_features=visual_size, momentum=momentum)

        self.W_v_h = nn.Linear(in_features=hidden_size, out_features=visual_size)
        self.bn_v_h = nn.BatchNorm1d(num_features=visual_size, momentum=momentum)

        self.W_v_att = nn.Linear(in_features=visual_size, out_features=visual_size)
        self.bn_v_att = nn.BatchNorm1d(num_features=visual_size, momentum=momentum)

        self.W_a = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.bn_a = nn.BatchNorm1d(num_features=k, momentum=momentum)

        self.W_a_h = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.bn_a_h = nn.BatchNorm1d(num_features=1, momentum=momentum)

        self.W_a_att = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.bn_a_att = nn.BatchNorm1d(num_features=k, momentum=momentum)

        # self.W_fc = nn.Linear(in_features=visual_size, out_features=embed_size)  # for v3
        self.W_fc = nn.Linear(in_features=visual_size , out_features=embed_size)
        self.bn_fc = nn.BatchNorm1d(num_features=embed_size, momentum=momentum)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

        self.__init_weight()

    def __init_weight(self):
        self.W_v.weight.data.uniform_(-0.1, 0.1)
        self.W_v.bias.data.fill_(0)

        self.W_v_h.weight.data.uniform_(-0.1, 0.1)
        self.W_v_h.bias.data.fill_(0)

        self.W_v_att.weight.data.uniform_(-0.1, 0.1)
        self.W_v_att.bias.data.fill_(0)

        self.W_a.weight.data.uniform_(-0.1, 0.1)
        self.W_a.bias.data.fill_(0)

        self.W_a_h.weight.data.uniform_(-0.1, 0.1)
        self.W_a_h.bias.data.fill_(0)

        self.W_a_att.weight.data.uniform_(-0.1, 0.1)
        self.W_a_att.bias.data.fill_(0)

        self.W_fc.weight.data.uniform_(-0.1, 0.1)
        self.W_fc.bias.data.fill_(0)

    def forward(self, avg_features, h_sent):
        if self.version == 'v1':
            return self.v1(avg_features, h_sent)
        elif self.version == 'v4':
            return self.v4(avg_features, h_sent)

    def v1(self, avg_features, h_sent):
        """
        only training
        :rtype: object
        """
        W_v = self.bn_v(self.W_v(avg_features))
        W_v_h = self.bn_v_h(self.W_v_h(h_sent.squeeze(1)))

        alpha_v = self.softmax(self.bn_v_att(self.W_v_att(self.tanh(W_v + W_v_h))))
        v_att = torch.mul(alpha_v, avg_features)
        ctx = self.W_fc(v_att)

        return ctx, alpha_v

    def v4(self, avg_features, h_sent):
        W_v = self.W_v(avg_features)
        W_v_h = self.W_v_h(h_sent.squeeze(1))
        alpha_v = self.softmax(self.W_v_att(self.tanh(torch.add(W_v, W_v_h))))
        v_att = torch.mul(alpha_v, avg_features)
        ctx = self.W_fc(v_att)
        return ctx, alpha_v

class SemanticAttention(nn.Module):
    def __init__(self,
                 version='v1',
                 embed_size=512,  # 语义特征
                 hidden_size=512,  # 隐藏层
                 visual_size=2048,  # 视觉特征
                 k=10,
                 momentum=0.1):
        super(SemanticAttention, self).__init__()
        self.version = version
        self.W_v = nn.Linear(in_features=visual_size, out_features=visual_size)
        self.bn_v = nn.BatchNorm1d(num_features=visual_size, momentum=momentum)

        self.W_v_h = nn.Linear(in_features=hidden_size, out_features=visual_size)
        self.bn_v_h = nn.BatchNorm1d(num_features=visual_size, momentum=momentum)

        self.W_v_att = nn.Linear(in_features=visual_size, out_features=visual_size)
        self.bn_v_att = nn.BatchNorm1d(num_features=visual_size, momentum=momentum)

        self.W_a = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.bn_a = nn.BatchNorm1d(num_features=k, momentum=momentum)

        self.W_a_h = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.bn_a_h = nn.BatchNorm1d(num_features=1, momentum=momentum)

        self.W_a_att = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.bn_a_att = nn.BatchNorm1d(num_features=k, momentum=momentum)

        # self.W_fc = nn.Linear(in_features=visual_size, out_features=embed_size)  # for v3
        self.W_fc = nn.Linear(in_features=visual_size + hidden_size, out_features=embed_size)
        self.bn_fc = nn.BatchNorm1d(num_features=embed_size, momentum=momentum)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

        self.__init_weight()

    def __init_weight(self):
        self.W_v.weight.data.uniform_(-0.1, 0.1)
        self.W_v.bias.data.fill_(0)

        self.W_v_h.weight.data.uniform_(-0.1, 0.1)
        self.W_v_h.bias.data.fill_(0)

        self.W_v_att.weight.data.uniform_(-0.1, 0.1)
        self.W_v_att.bias.data.fill_(0)

        self.W_a.weight.data.uniform_(-0.1, 0.1)
        self.W_a.bias.data.fill_(0)

        self.W_a_h.weight.data.uniform_(-0.1, 0.1)
        self.W_a_h.bias.data.fill_(0)

        self.W_a_att.weight.data.uniform_(-0.1, 0.1)
        self.W_a_att.bias.data.fill_(0)

        self.W_fc.weight.data.uniform_(-0.1, 0.1)
        self.W_fc.bias.data.fill_(0)

    def forward(self, avg_features, semantic_features, h_sent):
        if self.version == 'v1':
            return self.v1(semantic_features, h_sent)
        elif self.version == 'v2':
            return self.v2(avg_features, semantic_features, h_sent)
        elif self.version == 'v3':
            return self.v3(avg_features, semantic_features, h_sent)
        elif self.version == 'v4':
            return self.v4(semantic_features, h_sent)
        elif self.version == 'v5':
            return self.v5(avg_features, semantic_features, h_sent)

    def v1(self, semantic_features, h_sent):
        """
        only training
        :rtype: object
        """
        W_a_h = self.bn_a_h(self.W_a_h(h_sent))
        W_a = self.bn_a(self.W_a(semantic_features))
        alpha_a = self.softmax(self.bn_a_att(self.W_a_att(self.tanh(torch.add(W_a_h, W_a)))))
        a_att = torch.mul(alpha_a, semantic_features).sum(1)

        ctx = self.W_fc(a_att)

        return ctx, alpha_a

    def v2(self, avg_features, semantic_features, h_sent) :
        """
        no bn
        :rtype: object
        """
        W_v = self.W_v(avg_features)
        W_v_h = self.W_v_h(h_sent.squeeze(1))

        alpha_v = self.softmax(self.W_v_att(self.tanh(W_v + W_v_h)))
        v_att = torch.mul(alpha_v, avg_features)

        W_a_h = self.W_a_h(h_sent)
        W_a = self.W_a(semantic_features)
        alpha_a = self.softmax(self.W_a_att(self.tanh(torch.add(W_a_h, W_a))))
        a_att = torch.mul(alpha_a, semantic_features).sum(1)

        ctx = self.W_fc(torch.cat([v_att, a_att], dim=1))

        return ctx, alpha_v, alpha_a

    def v3(self, avg_features, semantic_features, h_sent) :
        """

        :rtype: object
        """
        W_v = self.bn_v(self.W_v(avg_features))
        W_v_h = self.bn_v_h(self.W_v_h(h_sent.squeeze(1)))

        alpha_v = self.softmax(self.W_v_att(self.tanh(W_v + W_v_h)))
        v_att = torch.mul(alpha_v, avg_features)

        W_a_h = self.bn_a_h(self.W_a_h(h_sent))
        W_a = self.bn_a(self.W_a(semantic_features))
        alpha_a = self.softmax(self.W_a_att(self.tanh(torch.add(W_a_h, W_a))))
        a_att = torch.mul(alpha_a, semantic_features).sum(1)

        ctx = self.W_fc(torch.cat([v_att, a_att], dim=1))

        return ctx, alpha_v, alpha_a

    def v4(self,semantic_features, h_sent):

        W_a_h = self.W_a_h(h_sent)
        W_a = self.W_a(semantic_features)
        alpha_a = self.softmax(self.W_a_att(self.tanh(torch.add(W_a_h, W_a))))
        a_att = torch.mul(alpha_a, semantic_features).sum(1)

        ctx = self.W_fc(torch.cat([v_att, a_att], dim=1))

        return ctx, alpha_a

    def v5(self, avg_features, semantic_features, h_sent):
        W_v = self.W_v(avg_features)
        W_v_h = self.W_v_h(h_sent.squeeze(1))

        alpha_v = self.softmax(self.W_v_att(self.tanh(self.bn_v(torch.add(W_v, W_v_h)))))
        v_att = torch.mul(alpha_v, avg_features)

        W_a_h = self.W_a_h(h_sent)
        W_a = self.W_a(semantic_features)
        alpha_a = self.softmax(self.W_a_att(self.tanh(self.bn_a(torch.add(W_a_h, W_a)))))
        a_att = torch.mul(alpha_a, semantic_features).sum(1)

        ctx = self.W_fc(torch.cat([v_att, a_att], dim=1))

        return ctx, alpha_v, alpha_a

class AdaptiveAttention(nn.Module):
    def __init__(self,
                 version='v1',
                 embed_size=512,  # 语义特征
                 hidden_size=512,  # 隐藏层
                 visual_size=2048,  # 视觉特征
                 k=10,
                 momentum=0.1):
        super(CoAttention, self).__init__()
        self.version = version
        self.W_v = nn.Linear(in_features=visual_size, out_features=visual_size)
        self.bn_v = nn.BatchNorm1d(num_features=visual_size, momentum=momentum)

        self.W_v_h = nn.Linear(in_features=hidden_size, out_features=visual_size)
        self.bn_v_h = nn.BatchNorm1d(num_features=visual_size, momentum=momentum)

        self.W_v_att = nn.Linear(in_features=visual_size, out_features=visual_size)
        self.bn_v_att = nn.BatchNorm1d(num_features=visual_size, momentum=momentum)

        self.W_a = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.bn_a = nn.BatchNorm1d(num_features=k, momentum=momentum)

        self.W_a_h = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.bn_a_h = nn.BatchNorm1d(num_features=1, momentum=momentum)

        self.W_a_att = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.bn_a_att = nn.BatchNorm1d(num_features=k, momentum=momentum)

        # self.W_fc = nn.Linear(in_features=visual_size, out_features=embed_size)  # for v3
        self.W_fc = nn.Linear(in_features=visual_size + hidden_size, out_features=embed_size)
        self.bn_fc = nn.BatchNorm1d(num_features=embed_size, momentum=momentum)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

        self.__init_weight()

    def __init_weight(self):
        self.W_v.weight.data.uniform_(-0.1, 0.1)
        self.W_v.bias.data.fill_(0)

        self.W_v_h.weight.data.uniform_(-0.1, 0.1)
        self.W_v_h.bias.data.fill_(0)

        self.W_v_att.weight.data.uniform_(-0.1, 0.1)
        self.W_v_att.bias.data.fill_(0)

        self.W_a.weight.data.uniform_(-0.1, 0.1)
        self.W_a.bias.data.fill_(0)

        self.W_a_h.weight.data.uniform_(-0.1, 0.1)
        self.W_a_h.bias.data.fill_(0)

        self.W_a_att.weight.data.uniform_(-0.1, 0.1)
        self.W_a_att.bias.data.fill_(0)

        self.W_fc.weight.data.uniform_(-0.1, 0.1)
        self.W_fc.bias.data.fill_(0)

    def forward(self, avg_features, semantic_features, h_sent):
        if self.version == 'v1':
            return self.v1(avg_features, semantic_features, h_sent)
        elif self.version == 'v2':
            return self.v2(avg_features, semantic_features, h_sent)
        elif self.version == 'v3':
            return self.v3(avg_features, semantic_features, h_sent)
        elif self.version == 'v4':
            return self.v4(avg_features, semantic_features, h_sent)
        elif self.version == 'v5':
            return self.v5(avg_features, semantic_features, h_sent)

    def v1(self, avg_features, semantic_features, h_sent):
        """
        only training
        :rtype: object
        """
        W_v = self.bn_v(self.W_v(avg_features))
        W_v_h = self.bn_v_h(self.W_v_h(h_sent.squeeze(1)))

        alpha_v = self.softmax(self.bn_v_att(self.W_v_att(self.tanh(W_v + W_v_h))))
        v_att = torch.mul(alpha_v, avg_features)

        W_a_h = self.bn_a_h(self.W_a_h(h_sent))
        W_a = self.bn_a(self.W_a(semantic_features))
        alpha_a = self.softmax(self.bn_a_att(self.W_a_att(self.tanh(torch.add(W_a_h, W_a)))))
        a_att = torch.mul(alpha_a, semantic_features).sum(1)

        ctx = self.W_fc(torch.cat([v_att, a_att], dim=1))

        return ctx, alpha_v, alpha_a

    def v2(self, avg_features, semantic_features, h_sent) :
        """
        no bn
        :rtype: object
        """
        W_v = self.W_v(avg_features)
        W_v_h = self.W_v_h(h_sent.squeeze(1))

        alpha_v = self.softmax(self.W_v_att(self.tanh(W_v + W_v_h)))
        v_att = torch.mul(alpha_v, avg_features)

        W_a_h = self.W_a_h(h_sent)
        W_a = self.W_a(semantic_features)
        alpha_a = self.softmax(self.W_a_att(self.tanh(torch.add(W_a_h, W_a))))
        a_att = torch.mul(alpha_a, semantic_features).sum(1)

        ctx = self.W_fc(torch.cat([v_att, a_att], dim=1))

        return ctx, alpha_v, alpha_a

    def v3(self, avg_features, semantic_features, h_sent) :
        """

        :rtype: object
        """
        W_v = self.bn_v(self.W_v(avg_features))
        W_v_h = self.bn_v_h(self.W_v_h(h_sent.squeeze(1)))

        alpha_v = self.softmax(self.W_v_att(self.tanh(W_v + W_v_h)))
        v_att = torch.mul(alpha_v, avg_features)

        W_a_h = self.bn_a_h(self.W_a_h(h_sent))
        W_a = self.bn_a(self.W_a(semantic_features))
        alpha_a = self.softmax(self.W_a_att(self.tanh(torch.add(W_a_h, W_a))))
        a_att = torch.mul(alpha_a, semantic_features).sum(1)

        ctx = self.W_fc(torch.cat([v_att, a_att], dim=1))

        return ctx, alpha_v, alpha_a

    def v4(self, avg_features, semantic_features, h_sent):
        W_v = self.W_v(avg_features)
        W_v_h = self.W_v_h(h_sent.squeeze(1))

        alpha_v = self.softmax(self.W_v_att(self.tanh(torch.add(W_v, W_v_h))))
        v_att = torch.mul(alpha_v, avg_features)

        W_a_h = self.W_a_h(h_sent)
        W_a = self.W_a(semantic_features)
        alpha_a = self.softmax(self.W_a_att(self.tanh(torch.add(W_a_h, W_a))))
        a_att = torch.mul(alpha_a, semantic_features).sum(1)

        ctx = self.W_fc(torch.cat([v_att, a_att], dim=1))

        return ctx, alpha_v, alpha_a

    def v5(self, avg_features, semantic_features, h_sent):
        W_v = self.W_v(avg_features)
        W_v_h = self.W_v_h(h_sent.squeeze(1))

        alpha_v = self.softmax(self.W_v_att(self.tanh(self.bn_v(torch.add(W_v, W_v_h)))))
        v_att = torch.mul(alpha_v, avg_features)

        W_a_h = self.W_a_h(h_sent)
        W_a = self.W_a(semantic_features)
        alpha_a = self.softmax(self.W_a_att(self.tanh(self.bn_a(torch.add(W_a_h, W_a)))))
        a_att = torch.mul(alpha_a, semantic_features).sum(1)

        ctx = self.W_fc(torch.cat([v_att, a_att], dim=1))

        return ctx, alpha_v, alpha_a


class SentenceLSTM(nn.Module):
    def __init__(self,
                 embed_size=512,
                 hidden_size=512):
        super(SentenceLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, batch_first=True)
        self.W_h1 = nn.Linear(in_features=hidden_size, out_features=embed_size, bias=True)
        self.W_h2 = nn.Linear(in_features=hidden_size, out_features=embed_size, bias=True)
        self.W_v1 = nn.Linear(in_features=hidden_size, out_features=embed_size, bias=True)
        self.W_v2 = nn.Linear(in_features=hidden_size, out_features=embed_size, bias=True)
        self.W_1 = nn.Linear(in_features=hidden_size, out_features=1, bias=True)
        self.W_2 = nn.Linear(in_features=hidden_size, out_features=1, bias=True)

        self.W_ctx = nn.Linear(in_features=2 * embed_size, out_features=embed_size, bias=True)
        self.W_output = nn.Linear(in_features=hidden_size, out_features=embed_size, bias=True)
        self.W_stop = nn.Linear(in_features=hidden_size, out_features=2, bias=True)
        self.Wh = nn.Linear(in_features=embed_size, out_features=embed_size, bias=True)
        self.Wc = nn.Linear(in_features=embed_size, out_features=embed_size, bias=True)
        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.__init_weight()

    def __init_weight(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.W_h1.weight.data.uniform_(-0.1, 0.1)
        self.W_h1.bias.data.fill_(0)
        self.W_h2.weight.data.uniform_(-0.1, 0.1)
        self.W_h2.bias.data.fill_(0)
        self.W_v1.weight.data.uniform_(-0.1, 0.1)
        self.W_v1.bias.data.fill_(0)
        self.W_v2.weight.data.uniform_(-0.1, 0.1)
        self.W_v2.bias.data.fill_(0)
        self.W_1.weight.data.uniform_(-0.1, 0.1)
        self.W_1.bias.data.fill_(0)
        self.W_2.weight.data.uniform_(-0.1, 0.1)
        self.W_2.bias.data.fill_(0)
        self.W_ctx.weight.data.uniform_(-0.1, 0.1)
        self.W_ctx.bias.data.fill_(0)
        self.W_output.weight.data.uniform_(-0.1, 0.1)
        self.W_output.bias.data.fill_(0)
        self.W_stop.weight.data.uniform_(-0.1, 0.1)
        self.W_stop.bias.data.fill_(0)
        self.Wh.weight.data.uniform_(-0.1, 0.1)
        self.Wh.bias.data.fill_(0)
        self.Wc.weight.data.uniform_(-0.1, 0.1)
        self.Wc.bias.data.fill_(0)

    def forward(self, frontal, lateral, state, phid):

        h1 = self.W_h1(phid)  # [bs, 1, 512]
        h2 = self.W_h2(phid)
        v1 = self.W_v1(frontal)  # [bs, 49, 512]
        v2 = self.W_v2(lateral)

        joint_out1 = self.tanh(torch.add(v1, h1))  # [bs, 49, 512]
        joint_out2 = self.tanh(torch.add(v2, h2))
        join_output1 = self.W_1(joint_out1).squeeze(2)
        join_output2 = self.W_2(joint_out2).squeeze(2)  # [bs, 49]

        alpha_v1 = self.softmax(join_output1)
        alpha_v2 = self.softmax(join_output2)

        ctx1 = torch.sum(frontal * alpha_v1.unsqueeze(2), dim=1)
        ctx2 = torch.sum(lateral * alpha_v2.unsqueeze(2), dim=1)
        ctx = torch.cat((ctx1, ctx2), dim=1)  # [bs, 1024]
        ctx = self.W_ctx(ctx)

        output1, state_t1 = self.lstm(ctx.unsqueeze(1), state)  # [bs, 1, 512] [1, bs, 512]
        p_stop = self.tanh(self.W_stop(output1.squeeze(1)))  # 512->2

        output = self.W_output(output1.squeeze(1))
        topic = self.tanh(torch.add(ctx, output))
        # topic = self.tanh(torch.cat((ctx, output), dim=1))
        h0_word = self.tanh(self.Wh(self.dropout(topic))).unsqueeze(0)
        c0_word = self.tanh(self.Wc(self.dropout(topic))).unsqueeze(0)  # bs 1 512
        phid = output1
        # print(h0_word.shape)
        # sys.exit()
        return topic, p_stop, state_t1, h0_word, c0_word, phid


class WordLSTM(nn.Module):
    def __init__(self, embed_size,
                 hidden_size,
                 vocab_size,
                 n_max):
        super(WordLSTM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(p=0.2)
        self.n_max = n_max
        self.vocab_size = vocab_size
        self.__init_weights()

    def __init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def forward(self, caption, state):
        embeddings = self.embed(caption).unsqueeze(1)

        hidden, states_t = self.lstm(embeddings, state)
        output = self.linear(self.dropout(hidden[:, -1, :]))
        return output, states_t

    def sample(self, start_tokens, state):
        sampled_ids = np.zeros((np.shape(start_tokens)[0], self.n_max))  # BS n_max
        # print(sampled_ids.shape)
        sampled_ids[:, 0] = start_tokens.cpu().view(-1, )
        predicted = start_tokens
        for i in range(1, self.n_max):
            # if i==1:
            #     predicted = self.embed(predicted)
            # else:
            #     predicted = self.embed(predicted).unsqueeze(1)  # torch.Size([1, 1, 512])
            # print(state.shape)
            predicted = self.embed(predicted).unsqueeze(1)
            hidden, state = self.lstm(predicted, state)
            output = self.linear(self.dropout(hidden[:, -1, :]))  # 测试的时候有没有dropout无关了
            predicted = torch.max(output, 1)[1]
            sampled_ids[:, i] = predicted.cpu()
        return sampled_ids, predicted



# class GeneratorLoss(nn.Module):  # 对抗阶段使用
#     def __init__(self):
#         super(GeneratorLoss,self).__init__()
#
#     def forward(self, batch_size, seq_length, vocab_size, prediction, x, rewards):
#         x1 = x.view([-1,1]).long()
#
#         one_hot = torch.Tensor(batch_size * seq_length, vocab_size)
#         one_hot.zero_()
#         x2 = one_hot.scatter_(1, x1, 1)
#
#         pred1 = prediction.view([-1, batch_size])
#         pred2 = torch.log(torch.clamp(pred1, min=1e-20, max=1.0))
#         prod = torch.mul(x2, pred2)
#         reduced_prod = torch.sum(prod, dim=1)
#         rewards_prod = torch.mul(reduced_prod, rewards.view([-1]))
#         generator_loss = torch.sum(rewards_prod)
#         return generator_loss


if __name__ == '__main__':
    import torchvision.transforms as transforms

    import warnings
    warnings.filterwarnings("ignore")

    extractor = VisualFeatureExtractor(model_name='resnet152')
    mlc = MLC(fc_in_features=extractor.out_features)
    co_att = CoAttention(visual_size=extractor.out_features)
    sent_lstm = SentenceLSTM()
    word_lstm = WordLSTM(embed_size=512, hidden_size=512, vocab_size=2195, num_layers=1)

    images = torch.randn((4, 3, 224, 224))
    captions = torch.ones((4, 10)).long()
    hidden_state = torch.randn((4, 1, 512))

    # # image_file = '../data/images/CXR2814_IM-1239-1001.png'
#     # # images = Image.open(image_file).convert('RGB')
#     # # captions = torch.ones((1, 10)).long()
#     # # hidden_state = torch.randn((10, 512))
# #
# norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
#
# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.TenCrop(224),
#     transforms.Lambda(lambda crops: torch.stack([norm(transforms.ToTensor()(crop)) for crop in crops])),
# ])

# images = transform(images)
# images.unsqueeze_(0)
#
# # bs, ncrops, c, h, w = images.size()
# # images = images.view(-1, c, h, w)
#
    print("images:{}".format(images.shape))
    print("captions:{}".format(captions.shape))
    print("hidden_states:{}".format(hidden_state.shape))

    visual_features, avg_features = extractor.forward(images)

    print("visual_features:{}".format(visual_features.shape))
    print("avg features:{}".format(avg_features.shape))

    tags, semantic_features = mlc.forward(avg_features)

    print("tags:{}".format(tags.shape))
    print("semantic_features:{}".format(semantic_features.shape))

    ctx, alpht_v, alpht_a = co_att.forward(avg_features, semantic_features, hidden_state)

    print("ctx:{}".format(ctx.shape))
    print("alpht_v:{}".format(alpht_v.shape))
    print("alpht_a:{}".format(alpht_a.shape))

    topic, p_stop, hidden_state, states = sent_lstm.forward(ctx, hidden_state)
    # p_stop_avg = p_stop.view(bs, ncrops, -1).mean(1)

    print("Topic:{}".format(topic.shape))
    print("P_STOP:{}".format(p_stop.shape))
    # print("P_stop_avg:{}".format(p_stop_avg.shape))

    words = word_lstm.forward(topic, captions)
    print("words:{}".format(words.shape))

    cam = torch.mul(visual_features, alpht_v.view(alpht_v.shape[0], alpht_v.shape[1], 1, 1)).sum(1)
    cam.squeeze_()
    cam = cam.cpu().data.numpy()
    for i in range(cam.shape[0]):
        heatmap = cam[i]
        heatmap = heatmap / np.max(heatmap)
        print(heatmap.shape)
