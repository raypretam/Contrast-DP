import os 
from os import path
from parser.utils import Corpus, Vocab
from parser.utils.data import TextDataset, batchify
from parser.utils.dataset import UniversalDependenciesDatasetReader
import torch


def Contrast(config):
    if config.input_type == 'conllu':
        pos = UniversalDependenciesDatasetReader()
        pos.load(config.positive)
        # neg = UniversalDependenciesDatasetReader()
        # neg.load(config.negative)
    
    else:
        pos = Corpus.load(config.positive)
        # neg = Corpus.load(config.fnegative)

    # To-do for use_predicted
    if config.use_predicted:
        if config.input_type == "conllu":
            pos_predicted = UniversalDependenciesDatasetReader()
            pos_predicted.load(config.fpredicted_pos)
        else:
            pos_predicted = Corpus.load(config.fpredicted_pos)
                 
    vocab = Vocab.from_corpus(config=config, corpus=pos)   

    if config.use_predicted:
        pos_set = TextDataset(vocab.numericalize(pos, pos_predicted))
    else:
        pos_set = TextDataset(vocab.numericalize(pos))
    # neg_set = TextDataset(vocab.numericalize(neg))
    pos_loader = batchify(dataset=pos_set,
                        batch_size=config.batch_size,
                        n_buckets=config.buckets)
    # neg_loader, _ = batchify(dataset=neg_set,
    #                         batch_size=config.batch_size,
    #                         n_buckets=config.buckets)
    
    print(f"{'Positive:':6} {len(pos_set):5} sentences in total, "
            f"{len(pos_loader):3} batches provided")
    # print(f"{'Negative:':6} {len(neg_set):5} sentences in total, "
    #         f"{len(neg_loader):3} batches provided")
    
    return pos_loader