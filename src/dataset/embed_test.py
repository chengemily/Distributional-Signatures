import os
import itertools
import collections
import json
import pickle
from collections import defaultdict

import numpy as np
import torch
from torchtext.vocab import Vocab, Vectors

from embedding.avg import AVG
from embedding.cxtebd import CXTEBD
from embedding.wordebd import WORDEBD
import dataset.loader as loader
import dataset.stats as stats
from dataset.utils import tprint

def embed_terms(args, use_cache=True, path_to_json='ebd_cache.json'):
    """
    Embeds class strings into word representations.

    :param args
    :param path_to_json: (str) path to json file containing word embeddings

    :return: dict {newsgroup class (int id) : embedded vector (nparray of float)}
    """
    classes = [
        'mideast', 'space', 'sale', 'politics', 'graphics',
        'cryptography', 'windows', 'microsoft', 'guns',
        'religion', 'autos', 'medicine', 'mac', 'electronics',
        'hockey', 'atheism', 'motorcycles', 'pc', 'baseball', 'christian'
    ]
    if use_cache:
        with open('20news_reps_cache_.json') as json_file:
            return classes, json.load(json_file)

    # Not using cache: extract vectors from global set
    with open(path_to_json) as json_file:
        mappings = json.load(json_file)

    # Cache 20 news group reps
    cache = dict(zip(classes, [mappings[topic] for topic in classes]))
    with open('20news_reps_cache.json', 'w') as fp:
        json.dump(cache, fp)

def cache_word_reps(args, dest='ebd_cache.json'):
    """
    Caches word representations {str: list(float)} as json.
    :param args
    """
    path = args.word_vector
    cache = {}

    # Opening file
    file = open(path, 'r')
    count = 0

    # Build dict in memory
    for line in file:
        count += 1
        if count == 1: continue

        line_list = line.split(' ')
        cache[line_list[0]] = [float(elt) for elt in line_list[1:-1]]

    # Save cache
    with open(dest, 'w') as fp:
        json.dump(cache, fp)


def display_mappings(words):
    """
    Maps words into representation space and displays mappings.
    :param words: list(str) to map
    :return:
    """
