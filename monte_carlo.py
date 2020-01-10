import argparse
import collections
import math
import os
import sys

import matplotlib.pyplot as plt

from pwdmodels import model, pcfg, backoff, ngram_chain


def gen_guess_crack(estimations: [int], upper_bound):
    estimations.sort()
    gc_pairs = {}
    for idx, est in enumerate(estimations):
        if est < upper_bound:
            gc_pairs[est] = idx
        else:
            break
    return list(gc_pairs.keys()), list(gc_pairs.values())
    pass


def draw_guess_crack_curve(guess_crack_list, figure_filename):
    for name, guess_crack in guess_crack_list.items():
        guesses, cracked = guess_crack
        plt.plot(guesses, cracked, label=name)
        plt.xscale("log")
    plt.xlabel("Guesses")
    plt.ylabel("Cracked(%)")
    plt.grid(ls="--")
    plt.legend(loc=2)
    plt.savefig(figure_filename)

    pass


def single_pwd_set_multi_models():
    parser = argparse.ArgumentParser("Monte Carlo")
    parser.add_argument("-p", "--pwd-set", required=True, type=str, help="passwords file")
    parser.add_argument("-l", "--min-gram", required=False, type=int, default=2, help="minimum n-gram")
    parser.add_argument("-u", "--max-gram", required=False, type=int, default=5, help="maximum n-gram")
    parser.add_argument("-x", "--back-off-threshold", required=False, type=int, default=10,
                        help="threshold of back-off")
    parser.add_argument("-s", "--sample-size", required=False, type=int, default=10000,
                        help="sample size of monte carlo")
    parser.add_argument("-r", "--result", required=True, type=str, help="guess-crack will be put here")
    parser.add_argument("-t", "--test-set", required=True, type=str, help="test set")
    parser.add_argument("-f", "--figure", required=True, type=str,
                        help="guess-crack curve will be saved in this file")
    parser.add_argument("-n", "--monte-carlo-upper-bound", required=False, type=int, default=10 ** 20,
                        help="guess-number larger than this will be ignored")
    parser.add_argument("-m", "--model", required=True,
                        choices=['PCFG', 'back-off', '2-gram', '3-gram', '4-gram', '5-gram', '6-gram'], nargs='+',
                        help='select one or more models to use')
    args = parser.parse_args()
    if (not args.figure.endswith(".pdf")) and (not args.figure.endswith(".png")):
        print(f"error, --figure must ends with .pdf or .png")
        sys.exit(1)
    with open(args.pwd_set, 'rt') as f:
        training = [w.strip('\r\n') for w in f]
    models = {f"{i}-gram": ngram_chain.NGramModel(training, i) for i in range(args.min_gram, args.max_gram + 1)}

    print("n-gram")
    models["BackOff"] = backoff.BackoffModel(training, args.back_off_threshold)
    print("back-off")
    # models["PCFG"] = pcfg.PCFG(training)
    # print("pcfg")
    samples = {name: list(m.sample(args.sample_size)) for name, m in models.items()}
    print("samples")
    estimators = {name: model.PosEstimator(sample)
                  for name, sample in samples.items()}
    print("estimators")
    names = sorted(models.keys())
    fout = open(args.result, "w")
    splitter = " : "
    fout.write(f"{splitter.join(names)}\n")
    all_estimations = {name: [] for name in names}  # type: {str: []}
    test_set_size = 0
    with open(args.test_set, "r") as f:
        for pwd in f:
            test_set_size += 1
            pwd = pwd.strip("\r\n")
            estimations = [math.ceil(estimators[name].position(models[name].logprob(pwd)))
                           for name in names]
            for idx, name in enumerate(names):
                all_estimations[name].append(math.ceil(estimations[idx]))
            fout.write(splitter.join([f"{str(s)}" for s in estimations]))
            fout.write("\n")
    guess_crack_pairs = {name: "" for name in names}
    for name, all_estimation in all_estimations.items():
        guess_crack_pairs[name] = gen_guess_crack(all_estimation, args.monte_carlo_upper_bound)
    draw_guess_crack_curve(guess_crack_pairs, args.figure)

    fout.flush()
    fout.close()
    pass


def single_model_multi_pwd_sets():
    parser = argparse.ArgumentParser("Monte Carlo: Single model multi password sets")
    parser.add_argument("-p", "--pwd-sets", required=True, nargs="+", help="password sets")
    parser.add_argument("-m", "--model", required=True, choices=["backoff", "2", "3", "4", "5", "6", "pcfg"],
                        help="using specified model")
    parser.add_argument("-d", "--dict", required=False, type=str, default=None, help='dict for pcfg')
    parser.add_argument("-t", "--test-sets", required=True, nargs="+", help="test sets")
    parser.add_argument("-r", "--result", required=False, help="monte-carlo result, guess-number of test sets")
    parser.add_argument("-f", "--figure", required=True, help="guess-crack curve")
    parser.add_argument("-x", "--threshold", required=False, type=int, default=10, help="threshold for back-off model")
    parser.add_argument("-s", "--sample-size", required=False, type=int, default=10000,
                        help="sample size of monte carlo")
    parser.add_argument("-u", "--monte-carlo-upper-bound", required=False, type=int, default=10 ** 20,
                        help="upper bound of guess-number")
    parser.add_argument("-b", "--delta-for-2-gram", dest="delta2", required=False, type=float, default=.0,
                        help='additive delta smoothing for 2-gram')
    args = parser.parse_args()
    if len(args.pwd_sets) != len(args.test_sets):
        print("pwd sets does not match test sets")
        sys.exit(1)
    if (not args.figure.endswith(".pdf")) and (not args.figure.endswith(".png")):
        print(f"error, --figure must ends with .pdf or .png")
        sys.exit(1)
    guess_crack_pairs = {}
    dict4pcfg = collections.defaultdict(int)
    if args.dict is not None:
        with open(args.dict, "r") as fin:
            for line in fin:
                line = line.strip("\r\n")
                dict4pcfg[line] += 1
    else:
        dict4pcfg = None
    pass
    for pwd_set, test_set in zip(args.pwd_sets, args.test_sets):  # type: str, str
        print(f"processing: {pwd_set}, {test_set}")
        with open(pwd_set, 'rt') as f:
            training = [w.strip('\r\n') for w in f]
            model_choice = args.model
            if model_choice == "backoff":
                pwd_model = backoff.BackoffModel(training, args.threshold)
            elif model_choice == "2":
                pwd_model = ngram_chain.NGramModel(training, 2, delta=args.delta2)
            elif model_choice == "4":
                pwd_model = ngram_chain.NGramModel(training, 4)
            elif model_choice == 'pcfg':
                pwd_model = pcfg.PCFG(training, dict4pcfg)
            else:
                print("Model not found", file=sys.stderr)
                sys.exit(1)

        sample = list(pwd_model.sample(args.sample_size))
        print("sample done...")
        estimator = model.PosEstimator(sample)
        print("estimator done...")
        estimations = []
        with open(test_set, "r") as f_test:
            for pwd in f_test:
                pwd = pwd.strip("\r\n")
                estimation = estimator.position(pwd_model.logprob(pwd))
                estimations.append(math.ceil(estimation))
        print("test set processed...")
        guess_crack_pairs[test_set.split(os.sep)[-1] if os.sep in test_set else test_set] = \
            gen_guess_crack(estimations, args.monte_carlo_upper_bound)
    fout = open(args.result, "w")
    for gc_name, gc_pair in guess_crack_pairs.items():
        fout.write(f"{gc_name}\n")
        guesses, cracked = gc_pair
        fout.writelines([f"{g} : {c}\n" for g, c in zip(guesses, cracked)])
    fout.close()
    draw_guess_crack_curve(guess_crack_pairs, args.figure)
    pass


if __name__ == "__main__":
    # single_pwd_set_multi_models()
    single_model_multi_pwd_sets()
    pass
