import bisect
import collections
import itertools
import math
import os
import random
import re
import sys
from enum import Enum

import model
import numpy

sys.path.append(r"../utils")
import english_parse as eng_parse


class SubStruct(Enum):
    letter = 1
    digits = 2
    symbol = 3


def extract_patterns(pwd, class_dict):
    structure, groups = [], []
    pattern = re.compile("(?:([a-zA-Z]+)|([0-9]+)|[^a-zA-Z0-9]+)")
    matches = pattern.finditer(pwd)
    for match in matches:
        l_part, d_part = match.groups()
        group = match.group()
        group_len = len(group)
        if l_part:
            # is letter
            words = eng_parse.seg2(group.lower())
            idx = 0
            for word in words:
                word_len = len(word)
                word_before_lower = group[idx: idx + word_len]
                idx += word_len
                class_i = class_dict.get(word, -1)
                if class_i != -1:
                    structure.append((SubStruct.letter, class_i))
                    groups.append(word_before_lower)
            pass
        elif d_part:
            # is digits
            structure.append((SubStruct.digits, group_len))
            groups.append(group)
        else:
            structure.append((SubStruct.symbol, group_len))
            groups.append(group)
    return tuple(structure), groups


class SemanticWord2VecModel(model.Model):
    __class_number = 50
    __window = 3
    __separation = " "
    __special_chr = chr(3)
    __min_count = 1

    def __preprocess(self, model_dir, train, seg_file, cluster_file):
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        if not os.path.exists(seg_file):
            eng_parse.segmentation_from_list(train, seg_file, separation=self.__separation,
                                             window=self.__window - 1, special_chr=self.__special_chr)
            del train[:]
            pass
        if not os.path.exists(cluster_file):
            import word2vec as wv
            wv.word2clusters(train=seg_file, output=cluster_file,
                             classes=self.__class_number, min_count=self.__min_count, window=self.__window)

    def __read_dict(self, cluster_file, sep=None):
        if sep is None:
            sep = self.__separation
        __fin = open(cluster_file)
        class_dict = dict()
        for line in __fin:
            line = line.strip("\r\n")
            __word, __class = line.split(sep)
            class_dict[__word] = int(__class)
        return class_dict

    def __process_no_link(self, training_set, cluster_file, from_len=4, to_len=40,
                          sep=__separation, class_number=__class_number, ):
        # fin = open(training_set, "r")
        self.class_dict = self.__read_dict(cluster_file)

        def zero_dict():
            return collections.defaultdict(itertools.repeat(0).__next__)

        struct_dict = zero_dict()
        grammar_dict = collections.defaultdict(zero_dict)

        for line in training_set:
            line = line.strip("\r\n")
            if from_len > len(line) or to_len < len(line):
                continue
            structure, groups = extract_patterns(pwd=line, class_dict=self.class_dict)
            struct_dict[structure] += 1
            for pair, group in zip(structure, groups):
                grammar_dict[pair][group] += 1

        def process(dict__):
            keys = list(dict__.keys())
            cum_counts = numpy.array(list(dict__.values())).cumsum()
            return dict__, keys, cum_counts

        self.struct = process(struct_dict)
        self.grammar = {k: process(v) for k, v in grammar_dict.items()}

    def __init__(self, words, tag, with_counts=False, shelfname=None):
        __corpus = tag
        __model_dir = os.path.join("../media", __corpus)
        __seg_filename = os.path.join(__model_dir, "seg.txt")
        __cluster_filename = os.path.join(__model_dir, "cluster.txt")
        self.__preprocess(__model_dir, words, seg_file=__seg_filename, cluster_file=__cluster_filename)
        self.__process_no_link(words, cluster_file=__cluster_filename)
        pass

    def generate(self):

        def unpack(processed_dict):
            dict__, keys, cum_counts = processed_dict
            total = cum_counts[-1]
            idx = bisect.bisect_right(cum_counts, random.randrange(total))
            key = keys[idx]
            return -math.log2(dict__[key] / total), key

        log_prob_struct, struct = unpack(self.struct)
        log_prob = log_prob_struct
        res = ""
        for sub_struct_idx_pair in struct:
            try:
                log_prob_sub_struct, group = unpack(self.grammar[sub_struct_idx_pair])
            except KeyError:
                log_prob_sub_struct = 0
                group = ""
                pass
            log_prob += log_prob_sub_struct
            res += group
        return log_prob, res

    def logprob(self, pwd, leaveout=False):
        structure, groups = extract_patterns(pwd, self.class_dict)
        # TODO 对于不存在的 pair，返回 inf
        if len("".join(groups)) != len(pwd):
            return float("inf")
        structures, _, cum_sum = self.struct
        try:
            res = -math.log2((structures[structure] - leaveout) /
                             (cum_sum[-1] - leaveout))
        except (ZeroDivisionError, ValueError):
            return float("inf")
        assert res > 0
        grammar = self.grammar
        for pair, group in zip(structure, groups):
            try:
                terminals, _, cum_sum = grammar[pair]
            except KeyError:
                return float("inf")
            try:
                res -= math.log2((terminals[group] - leaveout) /
                                 (cum_sum[-1] - leaveout))
            except (ZeroDivisionError, ValueError):
                return float("inf")
        return res
