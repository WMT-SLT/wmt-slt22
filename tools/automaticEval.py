#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Automatic evaluation with BLEU, chrF++ and BLEURT for WMT-SLT 2022
    Confidence intervals obtained via bootstrap resampling
"""

import sys, os
import argparse
import re

import random
import numpy as np

from sacrebleu.metrics import BLEU, CHRF
from bleurt import score


def get_parser():
    '''
    Creates a new argument parser.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--iFolder',
                    required=False,
                    type=str,
                    default="../txt/", 
                    metavar="<inputFolder>",
                    help="Folder with the reference and system's outputs (txt format)" )
    parser.add_argument('-o', '--oFile',
                    required=False,
                    type=str,
                    default="../scores/automaticCHRF.csv", 
                    metavar="<inputFolder>",
                    help="Output file with BLEU, chrF++ and BLEURT automatic metrics (csv format)" )
    parser.add_argument('-b', '--bootstraps',
                    required=False,
                    type=int,
                    default=1000, 
                    metavar="<#bootstraps>",
                    help="Number of bootstrap samples for CI (default 1000)" )
    parser.add_argument('-c', '--BRTcheckpoint',
                    required=False,
                    type=str,
                    default="/home/cristinae/soft/bleurt/BLEURT-20", 
                    metavar="<BLEURT checkpoint>",
                    help="BLEURT checkpoint" )
    return parser



def ci_bs(distribution, n, confLevel):
    ''' Calculates confidence intervals for distribution at confLevel after the 
        generation of n boostrapped samples 
    '''

    bsScores = np.zeros(n)
    size = len(distribution)
    random.seed(16) 
    for i in range(0, n):
        # generate random numbers with repetitions, to extract the indexes of the sysScores array
        bootstrapedSys = np.array([distribution[random.randint(0,size-1)] for x in range(size)])
        # scores for all the bootstraped versions
        # this works because we assume the MT metric is calculated at sentence level
        bsScores[i] = np.mean(bootstrapedSys,0)

    # we assume distribution of the sample mean is normally distributed
    # number of bootstraps > 100
    mean = np.mean(bsScores,0)
    stdDev = np.std(bsScores,0,ddof=1)
    # Because it is a bootstraped distribution
    alpha = (100-confLevel)/2
    confidenceInterval = np.percentile(bsScores,[alpha,100-alpha])

    return (mean, mean-confidenceInterval[0])
    

def main(args=None):

    parser = get_parser()
    args = parser.parse_args(args)
 
    OUTfile = args.oFile
    BLEURTcheckpoint = args.BRTcheckpoint
    
    ''' Locate all the system outputs in the input folder'''
    nameINfolder=args.iFolder
    files = sorted(os.listdir(nameINfolder))

    ''' Locate the reference '''
    referenceFile = nameINfolder+'slttest2022.dsgs-de.dsgs-de.REFERENCE.txt'
    reference = [['']]*2
    with open(referenceFile, 'r') as file:
        reference[0]=file.read().split('\n')
    
    ''' Metric calculation for each submission '''
    bleu = BLEU()
#    chrf = CHRF(word_order=2)  #chrF++
    chrf = CHRF()  #chrF
    focusNews = 243 # line where focus news start in the txt files
    #outputHeader = '# submission, BLEUall, BLEUsrf, BLEUfn, chrF++all, chrF++srf, chrF++fn, BLEURTall, BLEURTsrf, BLEURTfn\n'
    outputHeader = '# submission, BLEUall, BLEUsrf, BLEUfn, chrFall, chrFsrf, chrFfn, BLEURTall, BLEURTsrf, BLEURTfn\n'
    output = open(OUTfile, 'w')
    output.write(outputHeader)
    for nameFile in files:
        if (nameFile.endswith('REFERENCE.txt')): 
           continue
        with open(nameINfolder+nameFile, 'r') as file:
           hypothesis=file.read().split('\n')
        team = re.search(r'slttest2022.dsgs-de.dsgs-de.(.*?)\.txt', nameFile).group(1)
        scores = team
        
        # BLEU calculation
        #print(bleu.corpus_score(hypothesis, reference, args.bootstraps)) # NOT WORKING
        res = bleu.corpus_score(hypothesis[0:-1], [reference[0][0:-1]], args.bootstraps) # stupid way to make it work
        value = re.search(r'\(μ = (.*?)\)',str(res)).group(1)
        scores = scores + ', ' + value
        res = bleu.corpus_score(hypothesis[0:focusNews-1], [reference[0][0:focusNews-1]], args.bootstraps)
        value = re.search(r'\(μ = (.*?)\)',str(res)).group(1)
        scores = scores + ', ' + value
        res = bleu.corpus_score(hypothesis[focusNews:-1], [reference[0][focusNews:-1]], args.bootstraps)  # only FocusNews
        value = re.search(r'\(μ = (.*?)\)',str(res)).group(1)
        scores = scores + ', ' + value
        
        # CHRF++ calculation
        res = chrf.corpus_score(hypothesis[0:-1], [reference[0][0:-1]], args.bootstraps) 
        value = re.search(r'\(μ = (.*?)\)',str(res)).group(1)
        scores = scores + ', ' + value
        res = chrf.corpus_score(hypothesis[0:focusNews-1], [reference[0][0:focusNews-1]], args.bootstraps)
        value = re.search(r'\(μ = (.*?)\)',str(res)).group(1)
        scores = scores + ', ' + value
        res = chrf.corpus_score(hypothesis[focusNews:-1], [reference[0][focusNews:-1]], args.bootstraps)
        value = re.search(r'\(μ = (.*?)\)',str(res)).group(1)
        scores = scores + ', ' + value

        # BLEURT calculation
        confLevel = 95  #confidence forced to be 95 for compatibility with SacreBLEU
        precision = 3
        scorer = score.BleurtScorer(BLEURTcheckpoint)
        sentenceScores = scorer.score(references=reference[0][0:-1], candidates=hypothesis[0:-1])
        mean, interval = ci_bs(sentenceScores, args.bootstraps, confLevel)   
        value = str(np.around(mean, precision)) + ' ± ' + str(np.around(interval, precision))
        scores = scores + ', ' + value
        mean, interval = ci_bs(sentenceScores[0:focusNews-1], args.bootstraps, confLevel)   
        value = str(np.around(mean, precision)) + ' ± ' + str(np.around(interval, precision))
        scores = scores + ', ' + value
        mean, interval = ci_bs(sentenceScores[focusNews:-1], args.bootstraps, confLevel)   
        value = str(np.around(mean, precision)) + ' ± ' + str(np.around(interval, precision))
        scores = scores + ', ' + value
        
        output.write(scores+'\n')
        print(scores)
    close(output)    

        
        
if __name__ == "__main__":
   main()
