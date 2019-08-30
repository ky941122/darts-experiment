""" Utilities """
import os
import logging
import shutil
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle

device = torch.device("cuda")


class MyDataset(Dataset):

    def __init__(self, data_dir):

        self.sent_idx_list = []
        self.voice_embed_list = []
        self.word_num_per_sent_list = []
        self.sent_num_per_example_list = []
        self.label_list = []

        file_list = os.listdir(data_dir)
        file_list = [file for file in file_list if file.endswith(".pkl")]

        for pkl_file in file_list:
            file_path = os.path.join(data_dir, pkl_file)
            f = open(file_path, 'rb')
            course_datas = pickle.load(f)
            f.close()

            course_voice_embed, course_sent_word_idx, course_sent_word_num, course_sent_label, course_window_length = course_datas

            course_window_length = [[l] for l in course_window_length]

            self.voice_embed_list += course_voice_embed
            self.sent_idx_list += course_sent_word_idx
            self.word_num_per_sent_list += course_sent_word_num
            self.label_list += course_sent_label
            self.sent_num_per_example_list += course_window_length

        assert len(self.sent_idx_list) == len(self.voice_embed_list) == len(self.word_num_per_sent_list) == len(self.sent_num_per_example_list) == len(self.label_list)


    def __getitem__(self, index):

        sent_idx = self.sent_idx_list[index]
        voice_embed = self.voice_embed_list[index]
        word_num_per_sent = self.word_num_per_sent_list[index]
        sent_num_per_example = self.sent_num_per_example_list[index]
        label = self.label_list[index]

        return torch.LongTensor(sent_idx), torch.FloatTensor(voice_embed), \
               torch.LongTensor(word_num_per_sent), torch.LongTensor(sent_num_per_example),\
               torch.LongTensor(label)

    def __len__(self):
        return len(self.label_list)


class DataProvider:

    def __init__(self, batch_size, dataset, is_cuda):

        self.batch_size = batch_size
        self.dataset = dataset
        self.is_cuda = is_cuda

        self.dataiter = None
        self.iteration = 0
        self.epoch = 0

    def build(self):

        # print("Building DataIterator...")

        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, drop_last=True)
        self.dataiter = dataloader.__iter__()

    def next(self):

        if self.dataiter is None:
            self.build()
        try:
            batch = self.dataiter.next()
            self.iteration += 1

            if self.is_cuda:
                batch = [batch[i].to(device) for i in range(len(batch))]
            return batch
        except StopIteration:
            self.epoch += 1
            self.build()
            self.iteration += 1

            batch = self.dataiter.next()
            if self.is_cuda:
                batch = [batch[i].to(device) for i in range(len(batch))]
            return batch


def get_logger(file_path):
    """ Make python logger """
    # [!] Since tensorboardX use default logger (e.g. logging.info()), we should use custom logger
    logger = logging.getLogger('darts')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


def param_size(model):
    """ Compute parameter size in MB """
    n_params = sum(
        np.prod(v.size()) for k, v in model.named_parameters() if not k.startswith('aux_head'))
    return n_params / 1024. / 1024.


class AverageMeter():
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res


def save_checkpoint(state, ckpt_dir, is_best=False):
    filename = os.path.join(ckpt_dir, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(ckpt_dir, 'best.pth.tar')
        shutil.copyfile(filename, best_filename)
