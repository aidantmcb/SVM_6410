#################################################################################################
########################################## svm_summaryplots_C.py ##########################################
# This is a secondary file for this project. 
# Purpose: plot and explore characteristic behavior of various plotted models, this time for the models produced
# during the parameter sweep of C.


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

f6file = '/Users/aidanmcbride/Documents/Sagitta-Runaways/final6age.fits'

X_train_sample, y_train_sample, X_test, y_test = getDataset(rebalance = False)

true_test = X_test[np.where(y_test == 1)[0], :]
false_test = X_test[np.where(y_test == 0)[0], :]

#######

yso_weights = [0.7, 1.0, 2.0]
regularizeCs = np.arange(0.2, 4.1, 0.2)
save_figs = True


modeldir_base = 'Models/regularize_c/'

for i in range(len(yso_weights)):
    weight = yso_weights[i]
    modeldir = modeldir_base + '/w{}/'.format(str(round(weight, 1)))

    true_false = pickle.load(open(modeldir + 'truefalse_w{}.pickle'.format(str(round(weight, 1))), 'rb'))

    true_n = true_false[0]
    false_n = true_false[1]
    true_p = true_false[2]
    false_p = true_false[3]



    fig, ax = plt.subplots(figsize = (8,6))
    ax.plot(regularizeCs, true_p, label = r'$T_p$')
    ax.plot(regularizeCs, false_p, label = r'$F_p$')
    ax.plot(regularizeCs, false_n, label = r'$F_n$')
    ax.set_xlabel('C')
    ax.set_ylabel('N Stars Classified')
    ax.set_title('Classification Performance')
    ax.legend()
    plt.savefig(modeldir + '/Plots/TpFpRates.png') if save_figs else None
    plt.show()

    selectmod = 0.8
    precision = true_p / (true_p + false_p)
    recall = true_p / (true_p + false_n)
    fig, axs = plt.subplots(2,1, sharex = True, gridspec_kw={'height_ratios':[2,1]})
    axs[0].plot(regularizeCs, precision, label = r'Precision $\frac{T_p}{T_p + F_p}$', color = 'r')
    axs[0].plot(regularizeCs, recall, label = r'Recall $\frac{T_p}{T_p + F_n}$', color = 'g')
    ymin, ymax = axs[0].get_ylim()
    axs[0].plot([selectmod, selectmod], [ymin, ymax], linestyle = 'dashed')

    axs[0].set_ylabel('Precision & Recall')
    axs[0].legend(frameon = False, loc = 'lower right')
    axs[1].fill_between(regularizeCs, true_p / len(true_test), false_p / len(true_test), color = 'blue', label = 'True PMS')
    axs[1].fill_between(regularizeCs, false_p / len(true_test), color = 'orange', label = 'False PMS')
    ymin, ymax = axs[1].get_ylim()
    axs[1].plot([selectmod, selectmod], [ymin, ymax], linestyle = 'dashed')
    axs[1].set_xlabel('C')
    axs[1].set_ylabel('Fraction')
    axs[1].legend(frameon = False, loc = 'upper left')
    plt.savefig(modeldir + '/Plots/PR_Classified.png') if save_figs else None
    plt.show()


    fig, ax = plt.subplots(figsize = (8,6))
    points = ax.scatter(false_p  , true_p, c = regularizeCs)
    ax.set_xlabel('False Positive')
    ax.set_ylabel('True Positive')
    fig.colorbar(points, label = 'Model PMS Weight')
    plt.show()

    fig, ax = plt.subplots(figsize = (8,6))
    points = ax.scatter(false_p / (true_p + false_p) * 100 , true_p , c = regularizeCs)
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, np.nanpercentile(false_p / (true_p + false_p) * 100, 95) + 6)
    ax.set_xlabel('% Contamination')
    ax.set_ylabel('True Positive')
    fig.colorbar(points, label = 'Model PMS Weight')
    plt.show()

    fig, ax = plt.subplots(figsize = (8,6))
    points = ax.scatter(false_p / (true_p + false_p) * 100 , true_p / len(true_test) * 100, c = regularizeCs)
    ax.set_xlabel('% Contamination')
    ax.set_ylabel('% YSOs Recovered')
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, np.nanpercentile(false_p / (true_p + false_p) * 100, 95) + 6)
    # print(np.sort(false_p / (true_p + false_p) * 100))
    fig.colorbar(points, label = 'Model PMS Weight')
    plt.savefig(modeldir + '/Plots/PR_Classified.png') if save_figs else None
    plt.show()
