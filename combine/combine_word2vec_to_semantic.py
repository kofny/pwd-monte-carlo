import collections
import os
import shutil

from combine.indicator.indicator_factory import Indicator, IndicatorFactory
from pwdmodels.semantic_word2vec_optimal import SemanticModel, Struct


def combine_word2vec_to_semantic(semantic_model_dir, word2vec_model_dir, combine_model_dir,
                                 use_indicator: Indicator, _threshold: float, meet_and_stop: bool = True):
    """

    :param meet_and_stop: find the first set can be combined with and then stop
    :param _threshold:
    :param use_indicator:
    :param semantic_model_dir:
    :param word2vec_model_dir:
    :param combine_model_dir:
    :return:
    """
    if meet_and_stop:
        combine_model_dir = os.path.join(combine_model_dir,
                                         "%s-stop.%02d" % (use_indicator.name, int(_threshold * 100 + .5)))
    else:
        combine_model_dir = os.path.join(combine_model_dir,
                                         "%s-next.%02d" % (use_indicator.name, int(_threshold * 100 + .5)))
    if not os.path.exists(combine_model_dir):
        os.mkdir(combine_model_dir)
    _indicator = IndicatorFactory.build(use_indicator, _threshold)
    # copy semantic grammars to path of combined model
    shutil.copy(os.path.join(semantic_model_dir, "grammar.pickle"), os.path.join(combine_model_dir, "grammar.pickle"))
    shutil.copy(os.path.join(semantic_model_dir, "noun_treecut.pickle"),
                os.path.join(combine_model_dir, "noun_treecut.pickle"))
    shutil.copy(os.path.join(semantic_model_dir, "verb_treecut.pickle"),
                os.path.join(combine_model_dir, "verb_treecut.pickle"))
    shutil.copy(os.path.join(semantic_model_dir, "rules.txt"), os.path.join(combine_model_dir, "rules.txt"))
    if not os.path.exists(os.path.join(combine_model_dir, "nonterminals")):
        os.mkdir(os.path.join(combine_model_dir, "nonterminals"))
    # count the word so that probability can be re-computed
    word_cnt = collections.defaultdict(int)
    fin_cluster = open(os.path.join(word2vec_model_dir, "seg.txt"), "r")
    for line in fin_cluster:
        line = line.strip("\r\n")
        words = line.split(" ")
        for word in words:
            word_cnt[word] += 1
    # count the classification of semantic-guesser
    semantic_nonterminals: {str: {str: float or int}} = {}
    for root, dirs, files in os.walk(os.path.join(semantic_model_dir, "nonterminals")):
        for file in files:
            _path = os.path.join(root, file)
            semantic_nonterminals[file]: {str: int} = {}
            with open(_path, "r") as fin_semantic:
                for line in fin_semantic:
                    word, prob = line.split("\t")
                    # 这里不能设置 0 为默认值，因为 nonterminals 文件夹下还有 number, symbol，这些在 word_cnt 中是没有的
                    # set the default *prob*
                    # note that the sum of special#.txt or number#.txt is 1, therefor using prob is okay~
                    semantic_nonterminals[file][word] = word_cnt.get(word, float(prob))
                    pass
            pass

    # get the grammar of word2vec-guesser
    # note that *number* and *symbol* are also contained
    #
    grammar_dict, _ = SemanticModel("", word2vec_model_dir, init=False).load_pickle()
    un_combined_cnt = 0
    for classification in grammar_dict:
        if classification[0] != Struct.letter:
            continue
        grammars, _, _ = grammar_dict.get(classification)
        word2vec_set = set(grammars.keys())
        combined = False

        if meet_and_stop:
            max_key = None
            max_similarity = 0
            for key, nonterminal in semantic_nonterminals.items():
                semantic_set = set(nonterminal)
                cur_similarity = _indicator.similarity(semantic_set, word2vec_set)
                if cur_similarity > max_similarity:
                    max_similarity = cur_similarity
                    max_key = key
            if max_key is not None:
                combined = True
                semantic_nonterminals.get(max_key).update(grammars)
        else:
            for nonterminal in semantic_nonterminals:
                semantic_set = set(semantic_nonterminals.get(nonterminal))
                if _indicator.can_combine(semantic_set, word2vec_set):
                    semantic_nonterminals.get(nonterminal).update(grammars)
                    combined = True
        if not combined:
            un_combined_cnt += 1
            # print("un_combined_cnt: ", un_combined_cnt, ", classification: ", classification)
            for word, freq in grammars.items():  # type: str, int
                word_len = len(word)
                if word[0].isdigit():
                    word_class = "char%s.txt" % word_len
                elif word[0].isalpha():
                    word_class = "number%s.txt" % word_len
                else:
                    word_class = "special%s.txt" % word_len
                semantic_nonterminals.get(word_class)[word] = freq
            pass
    for classification in semantic_nonterminals:
        fout = open(os.path.join(combine_model_dir, "nonterminals", classification), "w")
        total = sum(semantic_nonterminals[classification].values())
        for word, freq in semantic_nonterminals[classification].items():
            fout.write("%s\t%f\n" % (word, freq / total))
        fout.flush()
        fout.close()
    print("un_combined_cnt: ", un_combined_cnt)
    return un_combined_cnt
    pass


semantic_14_255_dir = "/home/cw/Codes/Python/Chaunecy/fudan-monte-carlo-pwd/models/rockyou-semantic-14-255"
word2vec_14_255_dir = "/home/cw/Codes/Python/Chaunecy/fudan-monte-carlo-pwd/models/rockyou-word2vec-14-255"
combine_14_255_dir = "/home/cw/Codes/Python/Chaunecy/fudan-monte-carlo-pwd/models/rockyou-combine-14-255"

strategies = (True, False)
thresholds = (.05, .2, .35, .45, .5, .55, .65, .8, .95)

for indicator in Indicator:
    can_break = False
    for threshold in thresholds:
        for strategy in strategies:
            print(f"indicator: {indicator}, threshold: {threshold}, meet_and_stop: {strategy}")
            _un_combined_cnt = combine_word2vec_to_semantic(semantic_model_dir=semantic_14_255_dir,
                                                            word2vec_model_dir=word2vec_14_255_dir,
                                                            combine_model_dir=combine_14_255_dir,
                                                            use_indicator=indicator,
                                                            _threshold=threshold,
                                                            meet_and_stop=strategy)
            if strategy is False and _un_combined_cnt == 100:
                can_break = True
                break
        if can_break:
            break
