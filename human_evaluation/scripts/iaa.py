#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Script for the statistical analysis of the manual evaluation for
    the WMT-SLT 2022 shared task
    Date: 07.10.2022
    Author: cristinae, extended by lefterav
"""

import argparse
import csv
import sys
import warnings
import math
import numpy as np
import collections

import warnings


import pandas
import pandas as pd
from pandas.errors import SettingWithCopyWarning
import pingouin as pg

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

TARGET_LANGUAGE_ID = 'trglang'
LIKERT_SCALE = 100
DISCRETIZED_SCALE = 6
ACCEPTED_ANNOTATORS = [] #['sggdeu01', 'sggdeu03', 'sggdeu04']

def getAnnotatorsIDs(df, column):
    return (df[column].unique())


def getSystemsIDs(df, column):
    return (df[column].unique())


def fleiss_kappa(M):
    """
    https://gist.github.com/skylander86/65c442356377367e27e79ef1fed4adee
    See `Fleiss' Kappa <https://en.wikipedia.org/wiki/Fleiss%27_kappa>`_.
    :param M: a matrix of shape (:attr:`N`, :attr:`k`) where `N` is the number of subjects and `k` is the number of categories into which assignments are made. `M[i, j]` represent the number of raters who assigned the `i`th subject to the `j`th category.
    :type M: numpy matrix
    """
    N, k = M.shape  # N is # of items, k is # of categories
    n_annotators = float(max(M.sum(axis=1)))  # # of annotators
    # print(N,k,n_annotators)

    #    n_annotators = 2.0
    p = np.sum(M, axis=0) / (N * n_annotators)
    P = (np.sum(M * M, axis=1) - n_annotators) / (n_annotators * (n_annotators - 1))
    Pbar = np.sum(P) / N
    PbarE = np.sum(p * p)

    kappa = (Pbar - PbarE) / (1 - PbarE)
    stdErr = math.sqrt(Pbar * (1 - Pbar) / (N * (1 - PbarE) ** 2))

    return str(round(kappa, 2)) + '$\pm$' + str(round(stdErr, 2))


def nominal_metric(a, b):
    return a != b


def interval_metric(a, b):
    return (a - b) ** 2


def ratio_metric(a, b):
    return ((a - b) / (a + b)) ** 2


def krippendorff_alpha(data, metric=interval_metric, force_vecmath=False, convert_items=float, missing_items=None):
    '''
    https://github.com/grrrr/krippendorff-alpha/blob/master/krippendorff_alpha.py
    Calculate Krippendorff's alpha (inter-rater reliability):
    
    data is in the format
    [
        {unit1:value, unit2:value, ...},  # coder 1
        {unit1:value, unit3:value, ...},   # coder 2
        ...                            # more coders
    ]
    or 
    it is a sequence of (masked) sequences (list, numpy.array, numpy.ma.array, e.g.) with rows corresponding to coders and columns to items
    
    metric: function calculating the pairwise distance
    force_vecmath: force vector math for custom metrics (numpy required)
    convert_items: function for the type conversion of items (default: float)
    missing_items: indicator for missing items (default: None)
    '''

    # number of coders
    m = len(data)

    # set of constants identifying missing values
    if missing_items is None:
        maskitems = []
    else:
        maskitems = list(missing_items)
    if np is not None:
        maskitems.append(np.ma.masked_singleton)

    # convert input data to a dict of items
    units = {}
    for d in data:
        try:
            # try if d behaves as a dict
            diter = d.items()
        except AttributeError:
            # sequence assumed for d
            diter = enumerate(d)

        for it, g in diter:
            if g not in maskitems:
                try:
                    its = units[it]
                except KeyError:
                    its = []
                    units[it] = its
                its.append(convert_items(g))

    units = dict((it, d) for it, d in units.items() if len(d) > 1)  # units with pairable values
    n = sum(len(pv) for pv in units.values())  # number of pairable values

    if n == 0:
        raise ValueError("No items to compare.")

    np_metric = (np is not None) and ((metric in (interval_metric, nominal_metric, ratio_metric)) or force_vecmath)

    Do = 0.
    for grades in units.values():
        if np_metric:
            gr = np.asarray(grades)
            Du = sum(np.sum(metric(gr, gri)) for gri in gr)
        else:
            Du = sum(metric(gi, gj) for gi in grades for gj in grades)
        Do += Du / float(len(grades) - 1)
    Do /= float(n)

    if Do == 0:
        return 1.

    De = 0.
    for g1 in units.values():
        if np_metric:
            d1 = np.asarray(g1)
            for g2 in units.values():
                De += sum(np.sum(metric(d1, gj)) for gj in g2)
        else:
            for g2 in units.values():
                De += sum(metric(gi, gj) for gi in g1 for gj in g2)
    De /= float(n * (n - 1))

    return 1. - Do / De if (Do and De) else 1.


def call4Fleiss(df, uniqID, scale=LIKERT_SCALE, discretize=6, score_key='score'):
    if discretize > 0:
        df['score'] = (df['score'] / (LIKERT_SCALE / DISCRETIZED_SCALE)).round(0).astype(int)
        scale = DISCRETIZED_SCALE
        fleiss_score_key = 'discretized_score'
    else:
        scale = LIKERT_SCALE
        fleiss_score_key = 'score'
    sentences = df[uniqID].unique()
    num_sentences = len(sentences) - 1
    input4Fleiss = np.array([[0 for i in range(scale + 1)]] * num_sentences)
    # Sorting (with a stable algorithm) to later take into account that the same user might have evaluated the same sentence twice
    # For interannotator agreement we only consider the first instance
    df.sort_values(by=['username'], inplace=True, kind='mergesort', ascending=False)
    for i in range(num_sentences):
        scores = df.loc[df[uniqID] == i + 1][score_key]
        raters = np.array(df.loc[df[uniqID] == i + 1]['username'])
        # print(df.loc[df['sentence']==i+1])
        input4Fleiss[i] = [0 for k in range(scale + 1)]  # necessito aquesta linea i no entenc per que
        # some sentences were annotated twice, we only keep the first one per annotator
        ann = 'nobody'
        j = 0
        for score in scores:
            if (ann != raters[j]):
                # input4Fleiss[i][score-1] = input4Fleiss[i][score-1] + 1
                input4Fleiss[i][score] = input4Fleiss[i][score] + 1
            ann = raters[j]
            j += 1
    return fleiss_kappa(input4Fleiss)


def main(csv_filenames, csv_header_file, score_scale, discretize_scale):
    df = read_csv_data(csv_filenames, csv_header_file)

    print(f"size of dataset: {len(df)}")

    # the real annotator username is only the first 8 digits
    df['username'] = df['username'].str[:8]

    if ACCEPTED_ANNOTATORS:
        df = df[df['username'].isin(ACCEPTED_ANNOTATORS)]

    annotators = getAnnotatorsIDs(df, 'username')
    systems = getSystemsIDs(df, TARGET_LANGUAGE_ID)

    # the unique segment id is a concatenation of doc_id and doc:segment_it
    df['seg_id'] = df['documentid'].map(str) + ":" + df['itemid'].map(str)

    # Adding standardized scores
    completeDF = pd.DataFrame()
    for annotator in annotators:
        tmpDF = df[df['username'] == annotator]
        tmpDF['z-score'] = (tmpDF['score'] - tmpDF['score'].mean()) / tmpDF['score'].std()
        completeDF = pd.concat([completeDF, tmpDF])

    # print(completeDF)

    # mean and std for the raw scores and z-scores
    # notice that pandas use ddof=1
    # means = df.groupby(TARGET_LANGUAGE_ID)['score'].mean().round(decimals=1)
    # devs = df.groupby(TARGET_LANGUAGE_ID)['score'].std().round(decimals=1)
    # zmeans = completeDF.groupby(TARGET_LANGUAGE_ID)['z-score'].mean().round(decimals=1)
    # zdevs = completeDF.groupby(TARGET_LANGUAGE_ID)['z-score'].std().round(decimals=1)
    # print("z-scores,  raw scores")
    # print(zmeans.astype(str) + u"$\pm$" + zdevs.astype(str) + u"    " + means.astype(str) + u"$\pm$" + devs.astype(str))
    # print()

    df['score'] = df['score'].round().astype(int)
    # print(df['score'])

    print_intra_annotator_agreement(annotators, df, score_scale, discretize_scale)

    print_inter_annotator_agreement(df, discretize_scale, score_scale)


def print_inter_annotator_agreement(df, discretize_scale, score_scale):
    # Interannotator agreement
    # Let's generate uniq IDs for the sentences
    df['sentence'] = df['system'] + ":" + df['seg_id']
    df['sentence'] = df[['sentence']].apply(lambda x: (pd.factorize(x)[0] + 1))
    # print("Inter Annotator agreement (Fleiss kappa)")
    kappa_value = call4Fleiss(df, uniqID='sentence', scale=score_scale, discretize=discretize_scale)

    # 4Krippendorff alpha
    # str seems needed for K-alpha:
    df['sentence'] = df[['sentence']].apply(lambda x: (pd.factorize(x)[0] + 1)).astype(str)
    alpha_value = get_krippendorff_alpha(df, discretize_scale)

    # icc = pg.intraclass_corr(data=df, targets='sentence', raters='username', ratings='score',
    #                         )
    # icc_value = icc['ICC'][1].round(2)
    # icc_std = round((icc['CI95%'][1][1] - icc['CI95%'][1][0]) / 2, 3)
    print(f" {kappa_value} & {alpha_value}")


def get_krippendorff_alpha(df, discretize_scale):
    if discretize_scale > 0:
        #df['score'] = (df['score'] / (LIKERT_SCALE / DISCRETIZED_SCALE)).round(0).astype(int)
        input4Krippendorff = df.pivot_table(index='username', columns='sentence', values='score', aggfunc="first")
    # print("Inter Annotator agreement (Krippendorff alpha)")
    alpha_value = round(krippendorff_alpha(input4Krippendorff), 2)
    return alpha_value


def print_intra_annotator_agreement(annotators, df, score_scale, discretize_scale=6):
    df['sentenceIntra'] = df['system'] + ":" + df['seg_id']

    print("""\\begin{tabular}{crrr}
        \\toprule
        annotator & \\multicolumn{1}{c}{kappa} & \\multicolumn{1}{c}{ICC} & items \\\\
        \\midrule
        """)

    for annotator in annotators:
        tmpDF = df[df['username'] == annotator]
        duplicatesTMP = tmpDF[tmpDF.duplicated(['sentenceIntra'], keep='first')]
        duplicatesTMP.loc[duplicatesTMP['username'] == annotator, 'username'] = annotator + 'a'
        duplicatesDF = tmpDF[tmpDF.duplicated(['sentenceIntra'], keep='last')]
        duplicatesDF.loc[duplicatesDF['username'] == annotator, 'username'] = annotator + 'b'
        duplicatesDF = pd.concat([duplicatesDF, duplicatesTMP])
        duplicatesDF['sentenceIntra'] = duplicatesDF[['sentenceIntra']].apply(lambda x: (pd.factorize(x)[0] + 1))

        fleiss_kappa_value = call4Fleiss(duplicatesDF, uniqID='sentenceIntra', scale=score_scale,
                                         discretize=discretize_scale)
        #icc = pg.intraclass_corr(data=df, targets='sentenceIntra', raters='username', ratings='score',
        #                         nan_policy='omit')
        #icc_value = icc['ICC'][1].round(2)
        # icc_std = round((icc['CI95%'][1][1] - icc['CI95%'][1][0]) / 2, 3)
        print(f"{annotator} & {fleiss_kappa_value} & {len(duplicatesTMP)} \\\\")

    print("""\\bottomrule
    \\end{tabular}
    """)


def icc(df):
    pass

def read_csv_data(csv_filenames: list, csv_header_filename: str) -> pandas.DataFrame:
    """
    Function that receives data in the Appraise format from many csv files and returns a single populated dataframe.
    :param csv_filenames: A list of filenames (full path) to be processed. The filenames are headerless
    :param csv_header_filename:  The filename (full path) that includes a csv row with the names of the CSV columns
    :return: a dataframe containing the entire data.
    """
    # open the header file and get the first row in a list form
    with open(csv_header_filename) as header_file:
        header_reader = csv.reader(header_file)
        column_names = next(header_reader)
    # open one by one the csv files and concatenate them
    df_list = []
    for csv_filename in csv_filenames:
        this_df = pd.read_csv(csv_filename, names=column_names)
        df_list.append(this_df)
    df = pd.concat(df_list, axis=0, ignore_index=True)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute inter-annotator agreement of WMT campaign")
    parser.add_argument('--csv-files', type=str, action='append',
                        help="The files that contain the human annotations (Appraise format)")
    parser.add_argument('--csv-header-file', type=str,
                        help="A csv file containing the header, i.e. the description of the columns")
    parser.add_argument('--discretize-scale', type=int, default=6,
                        help="Likert scale to discretize the full score scale for the Fleiss correation")
    parser.add_argument('--score-scale', type=int, default=100,
                        help="N as the range from 0 to N that the scores have been assigned")
    args = parser.parse_args()

    main(args.csv_files, args.csv_header_file, args.score_scale, args.discretize_scale)
