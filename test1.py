import argparse
from tqdm.notebook import tqdm
import torch
from utils.models import *
from utils.dataset import *
from utils.metrics import compute_scores
from torch.autograd import Variable
import os


class CaptionSampler(object):
    def __init__(self, args):
        self.args = args
        self.vocab = self.__init_vocab()
        self.transform = self.__init_transform()
        self.test_data_loader = self.__init_data_loader(self.args.file_lits)
        self.load_model_path = os.path.join(self.args.model_dir, self.args.load_model_path)
        self.model_state_dict = self.__load_mode_state_dict()
        self.extractor = self.__init_visual_extractor()
        self.semantic = self._init_semantic_embedding()
        self.sentence_model = self.__init_sentence_model()
        self.word_model = self.__init_word_word()

    def __init_vocab(self):
        with open(self.args.vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        return vocab

    def __init_transform(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.resize, self.args.resize)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        return transform

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

    def __load_mode_state_dict(self):
        try:
            model_state_dict = torch.load(self.load_model_path)
            print("[Load Model {} Succeed!]  ".format(self.load_model_path))
            print("Load From Epoch {}".format(model_state_dict['epoch']))
            return model_state_dict
        except Exception as err:
            print("[Load Model Failed] {}".format(err))
            raise err

    def __init_visual_extractor(self):
        model = VisualFeatureExtractor(self.args.embed_size)
        if self.model_state_dict is not None:
            model.load_state_dict(self.model_state_dict['extractor'])
            print("Visual Extractor Loaded!")
        if self.args.cuda:
            model = model.cuda()
        return model

    def _init_semantic_embedding(self):
        model = SemanticEmbedding(embed_size=self.args.embed_size)
        if self.model_state_dict is not None:
            model.load_state_dict(self.model_state_dict['semantic'])
            print("Semantic Embedding Loaded!")
        if self.args.cuda:
            model = model.cuda()
        return model

    def __init_sentence_model(self):
        model = SentenceLSTM(embed_size=self.args.embed_size,
                             hidden_size=self.args.hidden_size)
        if self.model_state_dict is not None:
            model.load_state_dict(self.model_state_dict['sentence_model'])
            print("Sentence Model Loaded!")
        if self.args.cuda:
            model = model.cuda()
        return model

    def __init_word_word(self):
        model = WordLSTM(embed_size=self.args.embed_size,
                         hidden_size=self.args.hidden_size,
                         vocab_size=len(self.vocab),
                         n_max=self.args.n_max)
        if self.model_state_dict is not None:
            model.load_state_dict(self.model_state_dict['word_model'])
            print("Word Model Loaded!")
        if self.args.cuda:
            model = model.cuda()
        return model

    def _to_var(self, x, requires_grad=False):
        if self.args.cuda:
            x = x.cuda()
        return Variable(x, requires_grad=requires_grad)

    def __vec2sent(self, array):  # Test
        sampled_caption = []
        for word_id in array:
            word = self.vocab.get_word_by_id(word_id)
            if word == '<start>':
                continue
            if word == '<end>' or word == '<pad>':
                break
            sampled_caption.append(word)
        return ' '.join(sampled_caption)

    def generate(self):
        self.extractor.eval()
        self.sentence_model.eval()
        self.word_model.eval()
        self.semantic.eval()
        progress_bar = tqdm(self.test_data_loader, desc='Generating')
        results = {}
        for images, study, label, captions, _ in progress_bar:
        # for images1, images2, targets, prob, study in progress_bar:
            images_frontal = self._to_var(images, requires_grad=False)
            images_lateral = self._to_var(images, requires_grad=False)
            frontal, lateral, avg = self.extractor.forward(images_frontal, images_lateral)  # [8, 49, 512] [8, 512]
            state_c, state_h = self.semantic.forward(avg)  # [BS, 30]

            pred_sentences = {}
            real_sentences = {}

            for i in study:
                pred_sentences[i] = {}
                real_sentences[i] = {}

            state = (torch.unsqueeze(state_c, 0), torch.unsqueeze(state_h, 0))
            phid = torch.unsqueeze(state_h, 1)
            for sentence_index in range(self.args.s_max):
                _,p_stop, state, h0_word, c0_word, phid = self.sentence_model.forward(frontal, lateral, state, phid)
                p_stop = p_stop.squeeze(1)
                p_stop = torch.unsqueeze(torch.max(p_stop, 1)[1], 1)
                states_word = (c0_word, h0_word)
                start_tokens = np.zeros(images_frontal.shape[0])
                start_tokens[:] = self.vocab('<start>')
                start_tokens = self._to_var(torch.Tensor(start_tokens).long(), requires_grad=False)
                sampled_ids, _ = self.word_model.sample(start_tokens, states_word)
                sampled_ids = sampled_ids * p_stop.cpu().numpy()
                for id, array in zip(study, sampled_ids):
                    pred_sentences[id][sentence_index] = self.__vec2sent(array)

            for id, array in zip(study, captions):
                for i, sent in enumerate(array):
                    real_sentences[id][i] = self.__vec2sent(sent)

            for id in study:
                print(id)
                print('Pred Sent.{}'.format(pred_sentences[id]))
                print('Real Sent.{}'.format(real_sentences[id]))
                results[id] = {'Pred Sent': pred_sentences[id], 'Real Sent': real_sentences[id]}
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

        val_met = compute_scores({i: [gt] for i, gt in enumerate(gts)},
                                 {i: [re] for i, re in enumerate(res)})
        print(val_met)


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='./data/images',
                        help='the path for images')
    parser.add_argument('--caption_json', type=str, default='./data/new_data/captions.json',
                        help='path for captions')
    parser.add_argument('--vocab_path', type=str, default='./data/new_data/vocab.pkl',
                        help='path for vocabulary')
    # parser.add_argument('--data_dir', type=str, default='/data1/liuyuxue/Data/iu_xray/annotation.json', help='path for images')
    parser.add_argument('--file_lits', type=str, default='./data/new_data/val_data.txt',
                        help='the path for test file list')
    parser.add_argument('--model_dir', type=str, default='./report_v4_models/v4/', help='path of model')
    parser.add_argument('--load_model_path', type=str, default='train_best.pth.tar', help='path of trained model')
    parser.add_argument('--resize', type=int, default=224, help='size for resizing images')
    parser.add_argument('--embed_size', type=int, default=512)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--s_max', type=int, default=6)
    parser.add_argument('--n_max', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    sampler = CaptionSampler(args)
    sampler.generate()
