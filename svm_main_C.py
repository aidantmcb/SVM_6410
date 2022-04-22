#################################################################################################
########################################## svm_main_C.py ##########################################
# This is the second main file for this project. 
# Purpose: generate SVM models to the input dataset, testing the best model hyperparameters for
# C, the regularization parameter.
# Summary plots carried out in svm_summaryplots_C.py


### Imports ###
from platform import java_ver
import numpy as np 
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pickle
from os.path import exists

from svm_utils import plotFormatting
from svm_datasets import getDataset # function to get a conistent dataset
plotFormatting()

# scikit-learn imports
from sklearn.svm import SVC 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

orionfile = '/Users/aidanmcbride/Documents/Sagitta-Runaways/master_revised.fits'
f6file = '/Users/aidanmcbride/Documents/Sagitta-Runaways/final6age.fits'
sagittafile = '/Users/aidanmcbride/Documents/Sagitta-Runaways/final6age.fits'


### DATA LOADING AND CLEANUP ###
# We need a representative set of data to feed our model, and we need it to be
# cleaned up a bit before we use it.
X_train_sample, y_train_sample, X_test, y_test = getDataset(rebalance = 0)

true_test = X_test[np.where(y_test == 1)[0], :]
false_test = X_test[np.where(y_test == 0)[0], :]


### Model Parameter Searching: weights ###
# Another model parameter is C, the regularization parameter, which tells our model how much it can have points inside the margins
# of its support vector boundaries. From past experience, let's try three yso weights [0.7, 1, 2] and explore what happens when we 
# go through the parameter space of C
yso_weights = [0.7, 1.0, 2.0] #[0.7, 1.0, 2.0] #range of yso weights 
# regularizeCs = np.arange(0.2, 4.1, 0.2)
# regularizeCs = np.append(regularizeCs, [10, 50, 100, 500, 1000, 10000, 100000])
regularizeCs = np.round(np.logspace(-1, 5, 20), 1)
np.set_printoptions(suppress=True)

# We're going to save the number of true negatives, false negatives, true positives, and false positives
# per model weight to evaluate them all. Initialize empty arrays

print('TEST YSOs TRUE/FALSE:', len(true_test) / len(false_test))



### Generate models ###
# Here we'll generate or load in the SVM models while tuning our class weights. 
# We also generate labels and calculate the confusion matrix for the model outputs.
# We'll save this to one last

## File saving preferences
modeldir_base = 'Models/regularize_c/' # when using data pms/!pms ratio
overwrite = False # Flag in case we want to recalculate already existing models
plotcm = True # Flag in case we want to plot or save the confusion matrix
## REDO WITH plotcm TRUE

## Iterate over model weights, carry out modeling 
for i in range(len(yso_weights)): # iterate over model params
    weight = yso_weights[i]

    true_n = np.zeros(len(regularizeCs))
    false_n = np.zeros(len(regularizeCs))
    true_p = np.zeros(len(regularizeCs))
    false_p = np.zeros(len(regularizeCs))

    modeldir = modeldir_base + '/w{}/'.format(str(round(weight, 1)))

    with PdfPages(modeldir + 'Plots/Confusion_Regularize_w{}.pdf'.format(str(round(weight,1)))) as pdf: # if want to save to pdf
    # with np.errstate(divide = 'ignore'): # if don't want to save to pdf

        for j in range(len(regularizeCs)):
            paramC = regularizeCs[j]
            print('Weight:', weight, 'C:', paramC)


            fname = modeldir + 'model_w{w}_C{c}.pickle'.format(w=str(round(weight, 1)), c = str(round(paramC, 1)))


            if exists(fname) & (not overwrite):
                model = pickle.load(open(fname, 'rb')) # load existing model if it exists


            else:
                # model pipeline is: scaling to normalize all values on [-1,1], then a
                # configurable SVM with true-yso class weight as a parameter

                model = Pipeline([('scaler', StandardScaler()), ('SVC', SVC(kernel = 'rbf', C = paramC,
                            class_weight ={0: 1, 1: weight}))])
                model.fit(X_train_sample, y_train_sample) # fit model to training set
                pickle.dump(model, open(fname, 'wb')) # save model to a .pickle file



            # predict class labels for the test set with the current model
            y_predict = model.predict(X_test)


            # separate test set into predicted true/false labels
            true_pred = X_test[np.where(y_predict == 1)[0], :]
            false_pred = X_test[np.where(y_predict == 0)[0]]



            true_test = X_test[np.where(y_test == 1)[0], :]
            false_test = X_test[np.where(y_test == 0)[0], :]
            print('PREDICTED TRUE / FALSE:', len(true_pred)/len(false_pred))
            print('PREDICTED TRUE / REAL TRUE:', len(true_pred) / len(true_test))



            ## HR Diagram
            # fig, ax = plt.subplots(figsize = (8,6))
            # ax.scatter(ms[:, 1] - ms[:, 2], ms[:,0], c = 'grey', alpha = 0.5)
            # ax.scatter(yso[:, 1] - yso[:, 2], yso[:,0], c = 'k', alpha = 0.5 )
            # ymin, ymax = ax.get_ylim()
            # ax.set_ylim(ymax, ymin)
            # plt.show()



            ### Confusion Matrix ###
            # The confusion matrix is a way of quantifying and visualizing the classification.
            # Given our true labels and model predictions, it returns an array of shape
            #                       [[true_negative, false_positive]
            #                        [false_negative, true_positive ]]
            # So, a perfect classification is a diagonal matrix.
            cm = confusion_matrix(y_test, y_predict)



            ## Plot the confusion matrix ##
            if plotcm:
                disp = ConfusionMatrixDisplay(cm, display_labels = model.classes_)
                disp.plot()
                ax = disp.ax_
                fig = disp.figure_
                ax.set_title('Weight: ' + str(weight) + ', C: ' + str(paramC))
                if 'pdf' in locals(): # save to PdfPages document if it's open
                    pdf.savefig(fig)
                    plt.close()
                else: 
                    plt.show()


            # retain number of true negatives, false negatives, true positives, false postiives
            true_n[j] = cm[0, 0]
            false_n[j] = cm[1, 0]
            true_p[j] = cm[1, 1]
            false_p[j] = cm[0, 1]


        # Save 
        truefalse = np.array([true_n, false_n, true_p, false_p])
        if exists(modeldir + 'truefalse_w{}.pickle'.format(str(round(weight,1)))) & (not overwrite):
            print('')
        else:
            pickle.dump(truefalse, open(modeldir + 'truefalse_w{}.pickle'.format(str(round(weight,1))), 'wb'))
