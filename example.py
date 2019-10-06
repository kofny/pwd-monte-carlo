#!/usr/bin/env python3

# Copyright 2016 Symantec Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0

# standard library
import argparse
import csv
# internal imports
import itertools

import backoff
import model
import ngram_chain
import pcfg
# import semantic_word2vec
import semantic_word2vec_optimal

parser = argparse.ArgumentParser()
parser.add_argument('passwordfile', help='password training set')
parser.add_argument('--min_ngram', type=int, default=2,
                    help='minimum n for n-grams')
parser.add_argument('--max_ngram', type=int, default=5,
                    help='maximum n for n-grams')
parser.add_argument('--backoff_threshold', type=int, default=10,
                    help='threshold for backoff')
parser.add_argument('--samplesize', type=int, default=10000,
                    help='sample size for Monte Carlo model')
parser.add_argument("--tag", type=str, default="default",
                    help="a tag that identifies the model")
parser.add_argument("--test", type=str, help="password test set")
parser.add_argument("--result", type=str, help="write results into this file")
args = parser.parse_args()

with open(args.passwordfile, 'rt') as f:
    training = [w.strip('\r\n') for w in f]
models = {"Semantic-word2vec": semantic_word2vec_optimal.SemanticModel(args.passwordfile, args.tag)}
# models = {'{}-gram'.format(i): ngram_chain.NGramModel(training, i)
#           for i in range(args.min_ngram, args.max_ngram + 1)}
# models['Backoff'] = backoff.BackoffModel(training, 10)
# models['PCFG'] = pcfg.PCFG(training)

samples = {name: list(model.sample(args.samplesize))
           for name, model in models.items()}

estimators = {name: model.PosEstimator(sample)
              for name, sample in samples.items()}
modelnames = sorted(models)

f_out = open(args.result, "w")
csv_out = csv.writer(f_out)
csv_out.writerow(["pwd"] + modelnames)

with open(args.test, "r") as f:
    for pwd in f:
        pwd = pwd.strip("\r\n")
        estimations = [estimators[name].position(models[name].logprob(pwd))
                       for name in modelnames]
        csv_out.writerow([pwd] + estimations)

f_out.flush()
f_out.close()

with open(args.result, "r") as csv_file:
    reader = csv.DictReader(csv_file)
    semantic_word2vec_col = [int(float(row["Semantic-word2vec"])) + 1 for row in reader]
    semantic_word2vec_col.sort()
    guesses, cracked = [0], [0]
    guess_crack = open("../media/%s/guess-crack.txt" % args.tag, "w")
    for m, n in itertools.groupby(semantic_word2vec_col):
        guesses.append(m)
        cracked.append(cracked[-1] + len(list(n)))
        guess_crack.write("%d : %d\n" % (guesses[-1], cracked[-1]))
