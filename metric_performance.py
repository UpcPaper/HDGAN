# # -*- coding:UTF-8 -*-
# from pycocoevalcap.eval import calculate_metrics
# import numpy as np
# import json
# import argparse
# import re
#
# def create_dataset(array):
#     dataset = {'annotations': []}
#
#     for i, caption in enumerate(array):
#         dataset['annotations'].append({
#             'image_id': i,
#             'caption': caption
#         })
#     return dataset
#
#
# def load_json(json_file):
#     with open(json_file, 'r') as f:
#         data = json.load(f)
#     return data
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument('--result_path', type=str,
#                         default='./results/generate.json')
#     args = parser.parse_args()
#
#     test = load_json(args.result_path)  # test是加载的结果
#     datasetGTS = {'annotations': []}
#     datasetRES = {'annotations': []}
#
#     for i, image_id in enumerate(test):
#         array = []  # 数组或者列表?
# 	bad_words = ['<unk>']
#         for each in test[image_id]['Pred Sent']:
#             array.append(test[image_id]['Pred Sent'][each])
#         pred_sent = '. '.join(array)    # 预测的结果
# 	pred_sent = re.sub(' <unk> ','', pred_sent)
#         array = []
#         for each in test[image_id]['Real Sent']:  # 真实的结果
#             sent = test[image_id]['Real Sent'][each]
#             if len(sent) != 0:
#                 array.append(sent)
#         real_sent = '. '.join(array)  # 用点点连起来
# 	real_sent = re.sub(' <unk> ','', real_sent)
#         datasetGTS['annotations'].append({
#             'image_id': i,
#             'caption': real_sent
#         })
#         datasetRES['annotations'].append({
#             'image_id': i,
#             'caption': pred_sent
#         })
#     print("datasetGTS", datasetGTS)
#     rng = range(len(test))  # 0-482 图片编号
#     print(calculate_metrics(rng, datasetGTS, datasetRES))
# -*- coding:UTF-8 -*-
from pycocoevalcap1.eval import calculate_metrics
import numpy as np
import json
import argparse
import re

def create_dataset(array):
    dataset = {'annotations': []}

    for i, caption in enumerate(array):
        dataset['annotations'].append({
            'image_id': i,
            'caption': caption
        })
    return dataset


def load_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--result_path', type=str,
                        default='./all/results/2023-03-23 08:35:36generate_lstm_best.json')
    args = parser.parse_args()

    test = load_json(args.result_path)  # test是加载的结果
    datasetGTS = {'annotations': []}
    datasetRES = {'annotations': []}

    for i, image_id in enumerate(test):
        array = []  # 数组或者列表?
        bad_words = ['<unk>']
        for each in test[image_id]['Pred Sent']:
            array.append(test[image_id]['Pred Sent'][each])
        pred_sent = '. '.join(array)    # 预测的结果
        pred_sent = re.sub(' <unk> ','', pred_sent)
        array = []
        for each in test[image_id]['Real Sent']:  # 真实的结果
            sent = test[image_id]['Real Sent'][each]
            if len(sent) != 0:
                array.append(sent)
        real_sent = '. '.join(array)  # 用点点连起来
        real_sent = re.sub(' <unk> ','', real_sent)
        datasetGTS['annotations'].append({
            'image_id': i,
            'caption': real_sent
        })
        datasetRES['annotations'].append({
            'image_id': i,
            'caption': pred_sent
        })
    # print("datasetGTS", datasetGTS)
    rng = range(len(test))  # 0-482 图片编号
    print(calculate_metrics(rng, datasetGTS, datasetRES))
