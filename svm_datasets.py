#################################################################################################
########################################## svm_datasets.py ##########################################
# This is a secondary file for this project. 
# Purpose: consistently reproduce training and test datasets for given input data file.


import numpy as np
from astropy.io import fits

f6file = '/Users/aidanmcbride/Documents/Sagitta-Runaways/final6age.fits'
f6file = 'final6age.fits'

sagittafile = 'sagitta_edr3.fits'

def getDataset(fname = None, colnames = ('G', 'BP', 'RP', 'J', 'H', 'K', 'PARALLAX', 'pms'), 
    traintest = ('train_set', 'test_set'), pmsthresh = 0.9, rebalance = False, sampsize = 10000):
    np.random.seed(1)
    ### DATA LOADING AND CLEANUP ###
    # We need a representative set of data to feed our model, and we need it to be
    # cleaned up a bit before we use it.
    if fname == None: 
        fname = f6file # DEFAULT
    data = fits.open(fname)[1].data 

    # Set column names here, if they differ:
    g, bp, rp, j, h, k, parallax, pms_label = colnames # change colnames if they differ in your file


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
    if type(traintest) == tuple: # DEFAULT - presorted inputs
        trainlabel, testlabel = traintest
        # Data is already divided into train_set (80%) and test_set (10%) - get their indices
        train = np.where(data[trainlabel])[0]
        test = np.where(data[testlabel])[0]
    elif traintest == False:
        # Just return all data as X_train and y_train, no test set
        train = np.arange(len(data))
        test = np.array([])
        sampsize = len(data) # reset sample size to all data and return
    else: 
        # Sample data for 90% train, 10% test
        train = np.random.choice(len(data), size = int(len(data) * 0.9))
        test = np.random.choice(len(data), size = int(len(data) * 0.1))



    # X and y for the model training set; what the model learns on
    X_train = X[train, :]
    y_train = y[train]

    # X and y for the model test set; what the model is evaluated on
    X_test = X[test, :]
    y_test = y[test]


    ## Downsampling: 
    # The dataset is a little too big to run comfortably. Moreover, pms stars only represent about 6% of the dataset
    # If needed, we can change the ratio of classes in the training data.

    # note: for a normal file, y ~ [1, 0], but this also allows for probability-based classifiers
    pms_inds, ms_inds = (np.where(y_train >= pmsthresh)[0], np.where(y_train < pmsthresh)[0])
    len1 = len(pms_inds)
    if rebalance == False: # DEFAULT
        len2 = len(ms_inds) # when using data pms/!pms ratio
    else:
        len2 = len1 * rebalance # when forcing ratio of pms/!pms = 1
    # get indices of X_train and y_train to sample pms and ms stars according to our class size choices
    ind = np.concatenate([np.random.choice(pms_inds,size = len1), np.random.choice(ms_inds, size = len2)]) 

    # Downsample traininig input size to sampsize. default = 10000 
    samp = np.random.choice(len(ind), size = sampsize) # randomly sample viable indices
    sample_indices = ind[samp] # we'll use these indices in model training if neededs

    X_train_sample = X_train[sample_indices, :] 
    y_train_sample = y_train[sample_indices]

    return (X_train_sample, y_train_sample, X_test, y_test)