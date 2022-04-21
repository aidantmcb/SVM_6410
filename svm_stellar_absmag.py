import numpy as np 
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from astropy.table import Table
import pickle

from os.path import exists

from utils import plotFormatting
plotFormatting()

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
data = fits.open(f6file)[1].data 

# Set column names here, if they differ:
g, bp, rp, j, h, k, parallax = ('G', 'BP', 'RP', 'J', 'H', 'K', 'PARALLAX')
pms_label = 'pms'


## Initial data setup: we're going to run our model on absolute magnitudes from Gaia and 2MASSs
# Calculate absolute magnitudes of all photometric magnitudes
G = data[g] - 5 * (np.log10(1000/data[parallax]) - 1)
BP = data[bp] - 5 * (np.log10(1000/data[parallax]) - 1)
RP = data[rp] - 5 * (np.log10(1000/data[parallax]) - 1)
J = data[j] - 5 * (np.log10(1000/data[parallax]) - 1)
H = data[h] - 5 * (np.log10(1000/data[parallax]) - 1)
K = data[k] - 5 * (np.log10(1000/data[parallax]) - 1)


## Define our input array as all absolute magnitudes. Has shape [N_samples, N_features]
# (where here N_samples is the number of rows in the fits file and N_features is the number of photometric mags)
X = np.array([G, BP, RP, J, H, K]).T.astype(np.float64)


## Clean up the data
# sklearn gets mad when there's NaN values - lets remove them
clean = np.where(np.all(np.isnan(X) == False, axis = 1))[0]
data = data[clean]
X = X[clean, :]


## Define our labels
# create our labels from the cleaned dataset - the file has a column named by spms_label indicating whether or not
# each star is pre-main sequence (True) or not (False)
y = data[pms_label] 


## Split divide up data
# Data is already divided into train_set (80%) and test_set (10%) - get their indices
train = np.where(data['train_set'])[0]
test = np.where(data['test_set'])[0]

# X and y for the model training set; what the model learns on
X_train = X[train, :]
y_train = y[train]

# X and y for the model test set; what the model is evaluated on
X_test = X[test, :]
y_test = y[test]
true_test = X_test[np.where(y_test == 1)[0], :]
false_test = X_test[np.where(y_test == 0)[0], :]


## Downsampling: 
# The dataset is a little too big to run comfortably. Moreover, pms stars only represent about 6% of the dataset
# If needed, we can change the ratio of classes in the training data 
pms_inds, ms_inds = (np.where(y_train==1)[0], np.where(y_train==0)[0])
len1 = len(pms_inds)
len2 = len(ms_inds) # when using data pms/!pms ratio
# len2 = len1 # when forcing ratio of pms/!pms = 1

# get indices of X_train and y_train to sample pms and ms stars according to our class size choices
ind = np.concatenate([np.random.choice(pms_inds,size = len1), np.random.choice(ms_inds, size = len2)]) 

# Downsample traininig input size to n_samples 
sampsize = 10000
samp = np.random.choice(len(ind), size = sampsize) # randomly sample viable indices
sample_indices = ind[samp] # we'll use these indices in model training if neededs

X_train_sample = X_train[sample_indices, :]
y_train_sample = y_train[sample_indices]

### Model Parameter Searching: weights ###
# One model parameter for sklearn.SVC is model weights, which takes a model weight corresponding to each class
# We can use this to favor yso stars despite their relatively low sample proportion; 
# however may lead to more false positives. Let's tune the model by training a lot of SVCs with varying YSO class weights
yso_weights = np.arange(0.1, 5, 0.1) #range of yso weights 

# We're going to save the number of true negatives, false negatives, true positives, and false positives
# per model weight to evaluate them all. Initialize empty arrays
true_n = np.zeros(len(yso_weights))
false_n = np.zeros(len(yso_weights))
true_p = np.zeros(len(yso_weights))
false_p = np.zeros(len(yso_weights))
print('TEST YSOs TRUE/FALSE:', len(true_test) / len(false_test))



### Generate models ###
# Here we'll generate or load in the SVM models while tuning our class weights. 
# We also generate labels and calculate the confusion matrix for the model outputs.
# We'll save this to one last

## File saving preferences
modeldir = 'Models/ratio_innate/' # when using data pms/!pms ratio
# modeldir = 'Models/ratio_balanced/' # when forcing ratio of pms/!pms = 1
overwrite = False # Flag in case we want to recalculate already existing models
plotcm = True # Flag in case we want to plot or save the confusion matrix


## Iterate over model weights, carry out modeling 
with PdfPages(modeldir + 'Plots/Confusion.pdf') as pdf: # if want to save to pdf
    # with np.errstate(divide = 'ignore'): # if don't want to save to pdf
    for i in range(len(yso_weights)): # iterate over model params
        weight = yso_weights[i]
        print('Weight:', weight)
        fname = modeldir + 'model_w{}.pickle'.format(str(round(weight, 1)))

        if exists(fname) & (not overwrite):
            model = pickle.load(open(fname, 'rb')) # load existing model if it exists
        else:
            # model pipeline is: scaling to normalize all values on [-1,1], then a
            # configurable SVM with true-yso class weight as a parameter
            model = Pipeline([('scaler', StandardScaler()), ('SVC', SVC(kernel = 'rbf', probability = False,
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
            ax.set_title('Weight: ' + str(weight))
            if 'pdf' in locals(): # save to PdfPages document if it's open
                pdf.savefig(fig)
                plt.close()
            else: 
                plt.show()

        true_n[i] = cm[0, 0]
        false_n[i] = cm[1, 0]
        true_p[i] = cm[1, 1]
        false_p[i] = cm[0, 1]

truefalse = np.array([true_n, false_n, true_p, false_p])
if exists(modeldir + 'truefalse.pickle') & (not overwrite):
    print('Done')
else:
    pickle.dump(truefalse, open(modeldir + 'truefalse.pickle', 'wb'))