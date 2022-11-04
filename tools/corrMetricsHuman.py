import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

fileScores = '../scores/all3metricsHuman.csv'
data = pd.read_csv(fileScores)


bleu = data['BLEU']
chrf = data['chrF']
rt = data['BLEURT']
z = data['Z Score']
h = data['H Score']
r = data['R Score']


print('BLEU-chrF')
corr, _ = pearsonr(bleu, chrf)
print('Pearsons correlation: %.3f' % corr)
corr, _ = spearmanr(bleu, chrf)
print('Spearmans correlation: %.3f' % corr)

print('BLEU-BLEURT')
corr, _ = pearsonr(bleu, rt)
print('Pearsons correlation: %.3f' % corr)
corr, _ = spearmanr(bleu, rt)
print('Spearmans correlation: %.3f' % corr)

print('chrF-BLEURT')
corr, _ = pearsonr(chrf, rt)
print('Pearsons correlation: %.3f' % corr)
corr, _ = spearmanr(chrf, rt)
print('Spearmans correlation: %.3f' % corr)

print('chrF-human (Z Score)')
corr, _ = pearsonr(chrf, z)
print('Pearsons correlation: %.3f' % corr)
corr, _ = spearmanr(chrf, z)
print('Spearmans correlation: %.3f' % corr)

print('BLEU-human (Z Score)')
corr, _ = pearsonr(bleu, z)
print('Pearsons correlation: %.3f' % corr)
corr, _ = spearmanr(bleu, z)
print('Spearmans correlation: %.3f' % corr)

print('BLEURT-human (Z Score)')
corr, _ = pearsonr(rt, z)
print('Pearsons correlation: %.3f' % corr)
corr, _ = spearmanr(rt, z)
print('Spearmans correlation: %.3f' % corr)



