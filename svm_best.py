#################################################################################################
########################################## svm_best.py ##########################################
# This is a overview file for this project. 
# Runs the best model for



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


X_train_sample, y_train_sample, X_test, y_test = getDataset()

true_test = X_test[np.where(y_test == 1)[0], :]
false_test = X_test[np.where(y_test == 0)[0], :]

#######

save_figs = True
extend = True

yso_weights = [0.7, 1.0, 2.0]
regularizeCs = np.round(np.logspace(-1, 5, 20),1)

m1 = pickle.load(open('Models/regularize_c/w0.7/model_w0.7_C143.8.pickle', 'rb'))
y_predict = m1.predict(X_test)

cm = confusion_matrix(y_test, y_predict)


true_n = cm[0,0] # N true negative 
false_n = cm[1,0] # N false negative
true_p = cm[1,1] # N true postive
false_p = cm[0,1] # N false positive


precision = true_p / (true_p + false_p) # What proportion of retrieved items are relevant?
recall = true_p / (true_p + false_n) # What proportion of relevant items are retrieved?
falsepositive = false_p / (true_p + false_p) # What proportion of items are falsely retrieved?

print('-----------------------------------------------')
print('Test Set Precision:', precision)
print('Test Set Recall:', recall)
print('Test Set False-Positive Rate:', falsepositive)
print('-----------------------------------------------')

####### APPLY TO SOME NEW DATA

# y here is a probability array, not a binary class
X, y, X_EMPTY, y_EMPTY, data = getDataset(fname = 'sagitta_edr3.fits', traintest = False, 
                colnames = ('g', 'bp', 'rp', 'j', 'h', 'k', 'parallax', 'pms'), table = True)
print('Input data length:', len(X))

pms_threshhold = 0.85
y = (y > pms_threshhold).astype(int)

y_predict = model.predict(X)

cm = confusion_matrix(y, y_predict)
disp = ConfusionMatrixDisplay(cm, display_labels = model.classes_)
disp.plot()
plt.show()

true_n = cm[0,0] # N true negative 
false_n = cm[1,0] # N false negative
true_p = cm[1,1] # N true postive
false_p = cm[0,1] # N false positive

precision = true_p / (true_p + false_p) # What proportion of retrieved items are relevant?
recall = true_p / (true_p + false_n) # What proportion of relevant items are retrieved?
falsepositive = false_p / (true_p + false_p) # What proportion of items are falsely retrieved?

print('-----------------------------------------------')
print('Test Set Precision:', precision)
print('Test Set Recall:', recall)
print('Test Set False-Positive Rate:', falsepositive)
print('-----------------------------------------------')



