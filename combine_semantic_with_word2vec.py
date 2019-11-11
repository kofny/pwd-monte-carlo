import collections
import os
import pickle
import shutil

import numpy

from semantic_word2vec_optimal import SemanticModel, Struct


def combine_semantic_to_word2vec(semantic_model_dir, word2vec_model_dir, combine_model_dir):
    word2vec_cluster = {}
    shutil.copy(os.path.join(word2vec_model_dir, "seg.txt"), os.path.join(combine_model_dir, "seg.txt"))
    shutil.copy(os.path.join(word2vec_model_dir, "struct.pickle"), os.path.join(combine_model_dir, "struct.pickle"))
    shutil.copy(os.path.join(word2vec_model_dir, "cluster.txt"), os.path.join(combine_model_dir, "cluster.txt"))
    grammar_dict, _ = SemanticModel("", word2vec_model_dir, init=False).load_pickle()
    with open(os.path.join(word2vec_model_dir, "cluster.txt"), "r") as fin_cluster:
        for line in fin_cluster:
            line = line.strip("\r\n")
            word, str_class = line.split(" ")
            word2vec_cluster[word] = {"class_id": int(str_class), "num": 0}
    with open(os.path.join(word2vec_model_dir, "seg.txt"), "r") as fin_seg:
        for line in fin_seg:
            line = line.strip("\r\n")
            words = line.split(" ")
            for word in words:
                if word in word2vec_cluster:
                    word2vec_cluster[word]["num"] += 1
    for root, dirs, files in os.walk(os.path.join(semantic_model_dir, "nonterminals")):
        for file in files:
            overlap_with = collections.defaultdict(int)
            path = os.path.join(root, file)
            fin = open(path, "r")
            words = []
            for line in fin:
                word, prob = line.split("\t")
                words.append(word)
                # find one and break
                if word in word2vec_cluster:
                    overlap_with[word2vec_cluster[word]["class_id"]] += 1
            fin.close()
            if len(overlap_with) == 0:
                continue
            class_id = max(overlap_with, key=overlap_with.get)
            for word in words:
                grammars, _, _ = grammar_dict[(Struct.letter, class_id)]
                if word not in grammars:
                    grammars[word] = word2vec_cluster[word]["num"]
            grammars, _, _ = grammar_dict[(Struct.letter, class_id)]
            grammar_dict[(Struct.letter, class_id)] = (
                grammars, list(grammars.keys()), numpy.array(list(grammars.values())).cumsum())
    fout_grammar = open(os.path.join(combine_model_dir, "grammar.pickle"), "wb")
    pickle.dump(grammar_dict, fout_grammar)
    fout_grammar.close()
    pass


def combine_word2vec_to_semantic(semantic_model_dir, word2vec_model_dir, combine_model_dir):
    semantic_cluster = {}
    shutil.copy(os.path.join(semantic_model_dir, "grammar.pickle"), os.path.join(combine_model_dir, "grammar.pickle"))
    shutil.copy(os.path.join(semantic_model_dir, "noun_treecut.pickle"),
                os.path.join(combine_model_dir, "noun_treecut.pickle"))
    shutil.copy(os.path.join(semantic_model_dir, "verb_treecut.pickle"),
                os.path.join(combine_model_dir, "verb_treecut.pickle"))
    shutil.copy(os.path.join(semantic_model_dir, "rules.txt"), os.path.join(combine_model_dir, "rules.txt"))
    if not os.path.exists(os.path.join(combine_model_dir, "nonterminals")):
        os.mkdir(os.path.join(combine_model_dir, "nonterminals"))
    word_cnt = collections.defaultdict(int)
    fin_cluster = open(os.path.join(word2vec_model_dir, "seg.txt"), "r")
    for line in fin_cluster:
        line = line.strip("\r\n")
        words = line.split(" ")
        for word in words:
            word_cnt[word] += 1
    semantic_nonterminals = {}
    for root, dirs, files in os.walk(os.path.join(semantic_model_dir, "nonterminals")):
        for file in files:
            path = os.path.join(root, file)
            semantic_nonterminals[file] = {}
            with open(path, "r") as fin_semantic:
                for line in fin_semantic:
                    word, prob = line.split("\t")
                    semantic_cluster[word] = {"class_id": file, "num": word_cnt.get(word, 0)}
                    semantic_nonterminals[file][word] = word_cnt.get(word, 0)
                    pass
            pass
    grammar_dict, _ = SemanticModel("", word2vec_model_dir, init=False).load_pickle()

    for key in grammar_dict:
        grammars, _, _ = grammar_dict.get(key)
        overlap_with = collections.defaultdict(int)
        for word in grammars:
            if word in semantic_cluster:
                overlap_with[semantic_cluster.get(word)["class_id"]] += 1
        if len(overlap_with) == 0:
            continue
        class_id = max(overlap_with, key=overlap_with.get)
        for word in grammars.keys():
            semantic_nonterminals[class_id][word] = grammars.get(word)
    for id in semantic_nonterminals:
        fout = open(os.path.join(combine_model_dir, "nonterminals", id), "w")
        total = sum(semantic_nonterminals[id].values())
        for word, freq in semantic_nonterminals[id].items():
            fout.write("%s\t%f\n" % (word, freq / total))
        fout.flush()
        fout.close()
    pass


# combine_semantic_to_word2vec(
#     "/home/cw/Codes/Python/Chaunecy/fudan-monte-carlo-pwd/models/rockyou-semantic-01-255",
#     "/home/cw/Codes/Python/Chaunecy/fudan-monte-carlo-pwd/models/rockyou-word2vec-01-255",
#     "/home/cw/Codes/Python/Chaunecy/fudan-monte-carlo-pwd/models/rockyou-combine-01-255/semantic-to-word2vec")
# combine_word2vec_to_semantic(
#     "/home/cw/Codes/Python/Chaunecy/fudan-monte-carlo-pwd/models/rockyou-semantic-14-255",
#     "/home/cw/Codes/Python/Chaunecy/fudan-monte-carlo-pwd/models/rockyou-word2vec-14-255",
#     "/home/cw/Codes/Python/Chaunecy/fudan-monte-carlo-pwd/models/rockyou-combine-14-255/word2vec-to-semantic"
# )
combine_word2vec_to_semantic(
    "/home/cw/Codes/Python/Chaunecy/fudan-monte-carlo-pwd/models/rockyou-semantic-01-255",
    "/home/cw/Codes/Python/Chaunecy/fudan-monte-carlo-pwd/models/rockyou-word2vec-01-255",
    "/home/cw/Codes/Python/Chaunecy/fudan-monte-carlo-pwd/models/rockyou-combine-01-255/word2vec-to-semantic"
)
# haha = SemanticModel("f",
#                      model_name="/home/cw/Codes/Python/Chaunecy/fudan-monte-carlo-pwd/models/rockyou-combine-14-255"
#                                 "/semantic-to-word2vec",
#                      init=False)
# grammar_dict, struct_dict = haha.load_pickle()
# print(grammar_dict[(Struct.letter, 6)][0].get("craving"))
