#################################################################################################
########################################## svm_summaryplots_w.py ##########################################
# This is a secondary file for this project. 
# Purpose: plot and explore characteristic behavior of various plotted models.


import numpy as np 
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.table import Table
import pickle

from svm_datasets import getDataset
from svm_utils import plotFormatting
plotFormatting()

from sklearn.svm import SVC 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

orionfile = '/Users/aidanmcbride/Documents/Sagitta-Runaways/master_revised.fits'
f6file = '/Users/aidanmcbride/Documents/Sagitta-Runaways/final6age.fits'

X_train_sample, y_train_sample, X_test, y_test = getDataset(rebalance = False)

true_test = X_test[np.where(y_test == 1)[0], :]
false_test = X_test[np.where(y_test == 0)[0], :]

#######

yso_weights = np.arange(0.1, 5, 0.1)
modeldir = 'Models/ratio_innate/'
# modeldir = 'Models/ratio_balanced/'

true_false = pickle.load(open(modeldir + 'truefalse.pickle', 'rb'))

true_n = true_false[0]
false_n = true_false[1]
true_p = true_false[2]
false_p = true_false[3]

save_figs = True


fig, ax = plt.subplots(figsize = (8,6))
ax.plot(yso_weights, true_p, label = r'$T_p$')
ax.plot(yso_weights, false_p, label = r'$F_p$')
ax.plot(yso_weights, false_n, label = r'$F_n$')
ax.set_xlabel('Model YSO Weight')
ax.set_ylabel('N Stars Classified')
ax.set_title('Classification Performance')
ax.legend()
plt.savefig(modeldir + '/Plots/TpFpRates.png') if save_figs else None
plt.show()

selectmod = 0.8




precision = true_p / (true_p + false_p) # What proportion of retrieved items are relevant?
recall = true_p / (true_p + false_n) # What proportion of relevant items are retrieved?
falsepositive = false_p / (true_p + false_p) # What proportion of items are falsely retrieved?

fig, axs = plt.subplots(2,1, sharex = True, gridspec_kw={'height_ratios':[2,1]})
axs[0].plot(yso_weights, precision, label = r'Precision $\frac{T_p}{T_p + F_p}$', color = 'r')
axs[0].plot(yso_weights, recall, label = r'Recall $\frac{T_p}{T_p + F_n}$', color = 'g')
ymin, ymax = axs[0].get_ylim()
axs[0].plot([selectmod, selectmod], [ymin, ymax], linestyle = 'dashed')

axs[0].set_ylabel('Precision & Recall')
axs[0].legend(frameon = False, loc = 'lower right')
axs[1].fill_between(yso_weights, true_p / len(true_test) + false_p / len(true_test), false_p / len(true_test), color = 'blue', label = 'True PMS')
axs[1].fill_between(yso_weights, false_p / len(true_test), color = 'orange', label = 'False PMS')
ymin, ymax = axs[1].get_ylim()
axs[1].plot([selectmod, selectmod], [ymin, ymax], linestyle = 'dashed')
axs[1].set_xlabel('Model PMS Weight')
axs[1].set_ylabel('Fraction')
axs[1].legend(frameon = False, loc = 'upper left')
plt.savefig(modeldir + '/Plots/PR_Classified.png') if save_figs else None
plt.show()




fig, ax = plt.subplots(figsize = (8,6))
points = ax.scatter(falsepositive * 100 , recall * 100, c = yso_weights)
ax.set_xlabel('% Contamination')
ax.set_ylabel('% YSOs Recovered')
xmin, xmax = ax.get_xlim()
ax.set_xlim(xmin, np.nanpercentile(falsepositive * 100, 95) + 6)
# print(np.sort(false_p / (true_p + false_p) * 100))
fig.colorbar(points, label = 'Model PMS Weight')

# critera = (100 * recall) / ( 100 * falsepositive  + 1)
# best = np.nanargmax(critera)
# points = ax.scatter(false_p[best] / (true_p[best] + false_p[best]) * 100 , recall[best] * 100, marker = '*', s = 100)

plt.savefig(modeldir + '/Plots/proportion_curve.png') if save_figs else None
plt.show()

