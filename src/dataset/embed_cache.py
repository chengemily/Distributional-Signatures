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

def embed_terms(args, classes, dest, use_cache=True, path_to_json='ebd_cache.json'):
    """
    Embeds class strings into word representations.

    :param args
    :param classes: (list of str) topic classes
    :param dest: (str) path to destination file
    :param path_to_json: (str) path to json file containing word embeddings

    :return: dict {newsgroup class (int id) : embedded vector (nparray of float)}
    """
    if use_cache:
        with open(dest) as json_file:
            return classes, json.load(json_file)

    # Not using cache: extract vectors from global set
    with open(path_to_json) as json_file:
        mappings = json.load(json_file)
    input()
    input(mappings)
    # Cache topic reps
    cache = dict(zip(classes, [mappings[topic] for topic in classes]))
    with open(dest, 'w') as fp:
        json.dump(cache, fp)

def cache_word_reps(args, dest='ebd_cache.json'):
    """
    Creates ebd_cache.json on a one-time basis.
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

