import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import spearmanr, pearsonr

fileScores = '../scores/all3metrics.csv'
data = pd.read_csv(fileScores)


x = data['BLEU']
y = data['chrF']
z = data['BLEURT']/1000


fig= plt.figure(figsize=(5.8,5.4))
ax = Axes3D(fig, auto_add_to_figure=False)
ax.set_aspect('auto', adjustable=None, anchor='NW', share=False)

fig.add_axes(ax)
#ax.scatter(x,y,z)


b, a = np.polyfit(x, z, deg=1)
ax.plot(x, z, 'r*', zdir='y', zs=20)
ax.plot(x, a + b * x, color="r", lw=1, zdir='y', zs=20)

b, a = np.polyfit(y, z, deg=1)
ax.plot(y, z, 'go', zdir='x', zs=0)
ax.plot(y, a + b *y, color="g", lw=1, zdir='x', zs=0)

b, a = np.polyfit(x, y, deg=1)
ax.plot(x, y, 'k+', zdir='z', zs=0.05)
ax.plot(x, a + b * x, color="k", lw=1, zs=0.05)

ax.set_xlim([0, 0.6])
ax.set_ylim([0, 20])
ax.set_zlim([0.05,0.20])

ax.set_xlabel(' BLEU', fontsize = 12, color = 'black')
ax.set_ylabel(' chrF', fontsize = 12, color = 'black')
ax.set_zlabel(' BLEURT', fontsize = 12, color = 'black')



print('BLEU-chrF')
corr, _ = pearsonr(x, y)
print('Pearsons correlation: %.3f' % corr)
corr, _ = spearmanr(x, y)
print('Spearmans correlation: %.3f' % corr)

print('BLEU-BLEURT')
corr, _ = pearsonr(x, z)
print('Pearsons correlation: %.3f' % corr)
corr, _ = spearmanr(x, z)
print('Spearmans correlation: %.3f' % corr)

print('chrF-BLEURT')
corr, _ = pearsonr(y, z)
print('Pearsons correlation: %.3f' % corr)
corr, _ = spearmanr(y, z)
print('Spearmans correlation: %.3f' % corr)

#plt.show() 
plt.savefig('../scores/metricsAll.png') 
