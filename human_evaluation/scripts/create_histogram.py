import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import csv
import glob
import numpy as np
import seaborn as sns  # for nicer graphics

BINS = 6
SCORE_THRESHOLD = 12.24
EXCLUDE = ['translator-A']
cmap = plt.get_cmap('jet_r')


def read_scores(path):
    scores = defaultdict(lambda: defaultdict(list))
    filenames = glob.glob(path)
    print(filenames)
    for filename in filenames:
        with open(filename) as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                system = row[1].replace('dfki', 'DFKI-SLT')
                system = system.replace('njupt-mtt', 'NJUPT-MTT')
                system = system.replace('MSMunich', 'MSMUNICH')
                system = system.replace('UZH-test', 'UZH (baseline)')
                # system = system.replace('translator-A', 'HUMAN')
                segment_id = row[7] + ":" + row[2]
                score = int(row[6])
                if score > SCORE_THRESHOLD and system not in EXCLUDE:
                #if score > SCORE_THRESHOLD and system in EXCLUDE:
                    scores[system][segment_id].append(score)

    avg_scores = defaultdict(list)
    for system, scores_per_segment in scores.items():

        for segment, segment_scores in scores_per_segment.items():
            avg_scores[system].append(np.mean(segment_scores))
    print(avg_scores)
    return avg_scores


def create_histogram(scores):

    # plt.figure(figsize=(8, 6))
    # N = len(scores.keys())
    # for i, system in enumerate(scores.keys()):
    #     print(system, len(scores[system]))
    #     color = cmap(float(i) / N)
    #     n, x, _ = plt.hist(scores[system], bins=BINS, alpha=0.5, label=system, color=color)

    sns.set_style({'font.family': 'serif', 'font.serif': 'Free Serif'})
    sns.displot(scores, bins=BINS, element="step", multiple="stack")
    plt.savefig("figures/histogram_stacked.pdf")
    sns.displot(scores, bins=BINS, element="step")
    plt.savefig("figures/histogram_transparent.pdf")
    sns.displot(scores, kind="kde")
    plt.savefig("figures/density.pdf")

    # sns.histplot(data=scores, bins=BINS, stat='density', alpha=0.2, kde=True,
    #              element='step', linewidth=0.5,
    #              line_kws=dict(alpha=0.5, linewidth=1.5))

    # plt.savefig("histogram.pdf")
    # plt.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    scores = read_scores(sys.argv[1])
    create_histogram(scores)
