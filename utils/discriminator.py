# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torchvision
import numpy as np
from torch.autograd import Variable
import torchvision.models as models
from torch.nn import utils as nn_utils
from torch.utils.data import DataLoader
import torch.nn.functional as F
import wordfreq


# class Discriminator(nn.Module):
#
#     def __init__(self, vocab_size, input_size=50, hidden_size=100, num_class=2, num_layers=1):
#         super(Discriminator, self).__init__()
#
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.embedding = nn.Embedding(vocab_size, input_size)
#         self.vocab_size = vocab_size
#         # self.__init_weights()
#         self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.classifier = nn.Linear(hidden_size, num_class)  # 将最后一个的rnn使用全连接的到最后的输出结果
#
#         # self.classifier.weight = Parameter(torch.Tensor(hidden_size, num_class))
#         # self.classifier.bias = Parameter(torch.Tensor(num_class))
#
#         self.classifier.weight.data.uniform_(-0.1, 0.1)
#         self.classifier.bias.data.fill_(0)
#         self.embedding.weight.data.uniform_(-0.1, 0.1)
#
#     def forward(self, x, lengths):
#         x = self.embedding(x)
#         x_packed = nn.utils.rnn.pack_padded_sequence(input=x, lengths=lengths, batch_first=True)
#         out, _ = self.rnn(x_packed)
#         x_padded, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
#         x = x_padded.contiguous()
#         x = x.view(-1, x.shape[2])
#         out = self.classifier(x)  # 得到分类结果 116 ,2
#         out = F.sigmoid(out)  # 如果使用交叉上损失 就不需要使用softmax了...
#
#         return out

class Discriminator(nn.Module):
    def __init__(self, seq_length, vocab_size, emb_size, filter_size, num_filter, dropoutRate):
        super(Discriminator, self).__init__()
        self.filter_size = filter_size
        self.num_filter = num_filter
        self.seq_length = seq_length
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.convs = nn.ModuleList()
        for fsize, fnum in zip(self.filter_size, self.num_filter):
            # kernel_size = depth, height, width
            conv = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=fnum,
                          kernel_size=(emb_size,fsize),  # 倒换
                          padding=0, stride=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(seq_length - fsize + 1, 1), stride=1)
            )
            self.convs.append(conv)
        self.dropout = nn.Dropout(p=dropoutRate)
        self.fc = nn.Linear(sum(self.num_filter), 2)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        embeds = self.embedding(x)
        embeds = torch.unsqueeze(embeds, 3)  # -1
        xs = list()
        for i, conv in enumerate(self.convs):
            x0 = conv(embeds)  # 应该是【x,x,1,1】
            x0 = x0.view((x0.shape[0], x0.shape[1]))
            xs.append(x0)
        cats = torch.cat(xs, 1)
        dropout = F.relu(self.dropout(cats))
        fc = F.relu(self.fc(dropout))
        y_prob = self.softmax(fc)
        return y_prob


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    disc = Discriminator(vocab_size=2105, input_size=10, hidden_size=10)
    captions = torch.LongTensor(2, 5).random_(0, 10)
    print("first =", captions)
    captions = Variable(captions)
    print(" shape of captions:{}".format(captions.shape))
    out = disc.forward(captions)
    print("NN output =", out)







