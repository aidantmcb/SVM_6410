#################################################################################################
########################################## svm_summaryplots_C.py ##########################################
# This is a secondary file for this project. 
# Purpose: plot and explore characteristic behavior of various plotted models, this time for the models produced
# during the parameter sweep of C.


import numpy as np 
np.set_printoptions(suppress=True)
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


X_train_sample, y_train_sample, X_test, y_test = getDataset(rebalance = False)

true_test = X_test[np.where(y_test == 1)[0], :]
false_test = X_test[np.where(y_test == 0)[0], :]

#######

save_figs = True

yso_weights = [0.7, 1.0, 2.0]
regularizeCs = np.round(np.logspace(-1, 5, 20),1)



modeldir_base = 'Models/regularize_c/'

for i in range(len(yso_weights)):
    weight = round(yso_weights[i],1)
    modeldir = modeldir_base + 'w{}/'.format(str(round(weight, 1)))

     
    true_false = pickle.load(open(modeldir + 'truefalse_w{}.pickle'.format(str(round(weight, 1))), 'rb'))

    true_n = true_false[0]
    false_n = true_false[1]
    true_p = true_false[2]
    false_p = true_false[3]


    # fig, ax = plt.subplots(figsize = (8,6))
    # ax.plot(regularizeCs, true_p, label = r'$T_p$')
    # ax.plot(regularizeCs, false_p, label = r'$F_p$')
    # ax.plot(regularizeCs, false_n, label = r'$F_n$')
    # ax.set_xlabel('C')
    # ax.set_ylabel('N Stars Classified')
    # ax.set_title('Weight = {}'.format('weight'))
    # ax.legend()
    # plt.savefig(modeldir + '/Plots/TpFpRates_w{}.png'.format(str(weight))) if save_figs else None
    # plt.show()



    # Plot precision/recall curves, and then percentages recovered vs contam
    precision = true_p / (true_p + false_p) # What proportion of retrieved items are relevant?
    recall = true_p / (true_p + false_n) # What proportion of relevant items are retrieved?
    falsepositive = false_p / (true_p + false_p) # What proportion of items are falsely retrieved?


    fig, axs = plt.subplots(2,1, sharex = True, gridspec_kw={'height_ratios':[2,1]})
    axs[0].plot(regularizeCs, precision, label = r'Precision $\frac{T_p}{T_p + F_p}$', color = 'r')
    axs[0].plot(regularizeCs, recall, label = r'Recall $\frac{T_p}{T_p + F_n}$', color = 'g')
    ymin, ymax = axs[0].get_ylim()
    axs[0].set_title('Weight = {}'.format(str(weight)))

    axs[0].set_ylabel('Precision & Recall')
    axs[0].legend(frameon = False, loc = 'lower right')
    axs[1].fill_between(regularizeCs, true_p / len(true_test)  + false_p / len(true_test), false_p / len(true_test), color = 'blue', label = 'True PMS')
    axs[1].fill_between(regularizeCs, false_p / len(true_test), color = 'orange', label = 'False PMS')
    ymin, ymax = axs[1].get_ylim()
    axs[1].set_xlabel('C')
    # xmin, xmax = axs[1].get_xlim()
    # axs[1].set_xlim(xmin - 0.2 * (xmax - xmin), xmax)
    axs[1].set_xscale('log')
    axs[1].set_ylabel('Fraction')
    axs[1].legend(frameon = False, loc = 'upper left')
    plt.savefig(modeldir + '/Plots/PR_Classified_w{}.png'.format(str(weight))) if save_figs else None
    plt.show()




    fig, ax = plt.subplots(figsize = (8,6))
    points = ax.scatter(falsepositive * 100 , recall * 100, c = np.log10(regularizeCs))
    ax.set_xlabel('% Contamination')
    ax.set_ylabel('% YSOs Recovered')
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, np.nanpercentile(falsepositive * 100, 95) + 5)
    fig.colorbar(points, label = 'log(C)')
    ax.set_title('Weight = {}'.format(str(weight)))

    criteria = (100 * recall) / ( 100 * falsepositive  + 1)
    best = np.nanargmax(criteria)
    points = ax.scatter(false_p[best] / (true_p[best] + false_p[best]) * 100 , recall[best] * 100, marker = '*', s = 100)

    print(criteria[best])

    paramC = round(regularizeCs[best], 1)
    bestfile = modeldir + 'model_w{w}_C{c}.pickle'.format(w=str(round(weight, 1)), c = str(round(paramC, 1)))
    print(bestfile)

    plt.savefig(modeldir + '/Plots/proportion_curve_w{}.png'.format(str(weight))) if save_figs else None
    plt.show()