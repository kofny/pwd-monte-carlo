import bisect
import collections
import itertools
import math
import os
import pickle
import random
import re
from enum import Enum

import model
import word2vec as wv
import numpy
import wordsegment as word_seg

word_seg.load()


# import english_parse as eng_parse

class Struct(Enum):
    letter = 1
    digits = 2
    symbol = 3


# LETTER = "1"
# DIGITS = "2"
# SYMBOL = "3"


def read_cluster(cluster_file, sep):
    fin = open(cluster_file, "r")
    word_class_dict = dict()
    for line in fin:
        line = line.strip("\r\n")
        word, __class = line.split(sep)
        word_class_dict[word] = int(__class)
    fin.close()
    return word_class_dict


struct_pattern = re.compile(r"(?:([a-zA-Z]+)|([0-9]+)|[^a-zA-Z0-9]+)")


def word2vec(_seg_file, _cluster_file, _classes, _size, _window, _min_count, _cbow):
    """

    :param _seg_file:
    :param _cluster_file:
    :param _classes:
    :param _size:
    :param _window:
    :param _min_count:
    :param _cbow: cbow = 1 represents that using cbow strategy
    :return:
    """
    wv.word2clusters(_seg_file, _cluster_file, _classes, _size, _window, _min_count, cbow=_cbow)
    pass


def fasttext(_seg_file, _cluster_file, _classes, _size, _window, _min_count, _cbow, trained_model_path):
    import fasttext
    model = fasttext.load_model(trained_model_path)

    pass


def extract_patterns(pwd, word_class_dict):
    structure, groups = [], []
    matches = struct_pattern.finditer(pwd)
    for match in matches:
        l_part, d_part = match.groups()
        group = match.group()
        group_len = len(group)
        if l_part:
            words = word_seg.segment(group)
            idx = 0
            for word in words:
                word_len = len(word)
                word_before_lower = group[idx: idx + word_len]
                idx += word_len
                class_i = word_class_dict.get(word, -1)
                if class_i != -1:
                    structure.append((Struct.letter, class_i))
                    groups.append(word_before_lower)
                    continue
        elif d_part:
            structure.append((Struct.digits, group_len))
            groups.append(group)
        else:
            structure.append((Struct.symbol, group_len))
            groups.append(group)
    del matches
    return tuple(structure), tuple(groups)


class SemanticModel(model.Model):
    __class_number = 50
    __window_size = 3
    __separation = " "
    __special_chr = chr(3)
    __min_count = 1

    def __preprocess(self, _model_dir, _training, _seg_file, _cluster_file):
        if not os.path.exists(_model_dir):
            os.mkdir(_model_dir)
        if not os.path.exists(_seg_file):
            fin = open(_training, "r")
            fout = open(_seg_file, "w")
            pattern = re.compile(r"(?:([a-zA-Z]+))")
            counter = 0
            for line in fin:
                if counter % 5000 == 0:
                    fout.flush()
                    print("preprocessed: %d" % counter)
                counter += 1
                line = line.strip("\r\n")
                matches = pattern.finditer(line)
                results = []
                for match in matches:
                    if len(match.group()) > 0:
                        results.extend(word_seg.segment(match.group()))
                if len(results) > 0:
                    fout.write("%s\n" % " ".join(results))
            print("preprocess done! %d" % counter)
            fin.close()
            fout.flush()
            fout.close()
            # eng_parse.segmentation(_training, _seg_file, separation=self.__separation,
            #                        window=self.__window_size - 1, special_chr=self.__special_chr)
        if not os.path.exists(_cluster_file):
            wv.word2clusters(train=_seg_file, output=_cluster_file,
                             classes=self.__class_number, min_count=self.__min_count, window=self.__window_size)
        pass

    def __process_no_link(self, training):
        if os.path.exists(self.grammar_pickle) and os.path.exists(self.struct_pickle):
            self.grammar_dict = pickle.load(open(self.grammar_pickle, "rb"))
            self.struct_dict = pickle.load(open(self.struct_pickle, "rb"))
            return

        def zero_dict():
            return collections.defaultdict(itertools.repeat(0).__next__)

        struct_dict = zero_dict()
        grammar_dict = collections.defaultdict(zero_dict)

        fin = open(training, "r")
        for line in fin:
            line = line.strip("\r\n")
            struct, groups = extract_patterns(line, self.word_class_dict)
            struct_dict[struct] += 1
            for pair, group in zip(struct, groups):
                grammar_dict[pair][group] += 1
        fin.close()

        def __process(_dict):
            keys = list(_dict.keys())
            cum_counts = numpy.array(list(_dict.values())).cumsum()
            return _dict, keys, cum_counts

        self.struct_dict = __process(struct_dict)
        self.grammar_dict = {k: __process(v) for k, v in grammar_dict.items()}
        pickle.dump(self.struct_dict, open(self.struct_pickle, "wb"))
        pickle.dump(self.grammar_dict, open(self.grammar_pickle, "wb"))
        pass

    def __init__(self, training, model_name, class_number=100, with_counts=False, init=True):
        model_dir = model_name
        seg_file = os.path.join(model_dir, "seg.txt")
        cluster_file = os.path.join(model_dir, "cluster.txt")
        self.__class_number = class_number
        self.struct_pickle = os.path.join(model_dir, "struct.pickle")
        self.grammar_pickle = os.path.join(model_dir, "grammar.pickle")
        if not init:
            return
        self.__preprocess(model_dir, training, seg_file, cluster_file)
        print("preprocess done")
        self.word_class_dict = read_cluster(cluster_file, self.__separation)
        self.__process_no_link(training)
        print("init done")

    def load_pickle(self):
        fin_struct = open(self.struct_pickle, "rb")
        struct_dict = pickle.load(fin_struct)
        fin_struct.close()
        fin_grammar = open(self.grammar_pickle, "rb")
        grammar_dict = pickle.load(fin_grammar)
        fin_grammar.close()
        return grammar_dict, struct_dict
        pass

    def generate(self):

        def unpack(packed_dict):
            _dict, keys, cum_counts = packed_dict
            total = cum_counts[-1]
            idx = bisect.bisect_right(cum_counts, random.randrange(total))
            key = keys[idx]
            return -math.log2(_dict[key] / total), key

        log_prob_struct, struct = unpack(self.struct_dict)
        log_prob = log_prob_struct
        res = ""
        for sub_struct_idx_pair in struct:
            try:
                log_prob_sub_struct, group = unpack(self.grammar_dict[sub_struct_idx_pair])
            except KeyError:
                log_prob_sub_struct = 0
                group = ""
                pass
            log_prob += log_prob_sub_struct
            res += group
        return log_prob, res
        pass

    def logprob(self, pwd, leaveout=False):
        struct, groups = extract_patterns(pwd, self.word_class_dict)
        if len("".join(groups)) != len(pwd):
            return float("inf")
        structures, _, cum_sum = self.struct_dict
        try:
            res = -math.log2((structures[struct] - leaveout) /
                             (cum_sum[-1] - leaveout))
        except (ZeroDivisionError, ValueError):
            return float("inf")
        assert res > 0
        grammar = self.grammar_dict
        for pair, group in zip(struct, groups):
            try:
                terminals, _, cum_sum = grammar[pair]
            except KeyError:
                return float("inf")
            try:
                res -= math.log2((terminals[group] - leaveout) /
                                 (cum_sum[-1] - leaveout))
            except(ZeroDivisionError, ValueError):
                return float("inf")
        return res
        pass
