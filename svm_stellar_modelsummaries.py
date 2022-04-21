import numpy as np 
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.table import Table
import pickle

from utils import plotFormatting
plotFormatting()

from sklearn.svm import SVC 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

orionfile = '/Users/aidanmcbride/Documents/Sagitta-Runaways/master_revised.fits'
f6file = '/Users/aidanmcbride/Documents/Sagitta-Runaways/final6age.fits'

data = fits.open(f6file)[1].data

# Set column names here, if they differ:
g, bp, rp, j, h, k, parallax = ('G', 'BP', 'RP', 'J', 'H', 'K', 'PARALLAX')

G = data[g] - 5 * (np.log10(1000/data[parallax]) - 1)
BP = data[bp] - 5 * (np.log10(1000/data[parallax]) - 1)
RP = data[rp] - 5 * (np.log10(1000/data[parallax]) - 1)
J = data[j] - 5 * (np.log10(1000/data[parallax]) - 1)
H = data[h] - 5 * (np.log10(1000/data[parallax]) - 1)
K = data[k] - 5 * (np.log10(1000/data[parallax]) - 1)
X = np.array([G, BP, RP, J, H, K]).T.astype(np.float64)

mask = np.where(np.all(np.isnan(X) == False, axis = 1))[0]

data = data[mask]
X = X[mask, :]
y = data['pms']

train = np.where(data['train_set'])[0]
test = np.where(data['test_set'])[0]

X_train = X[train, :]
y_train = y[train]

X_test = X[test, :]
y_test = y[test]

yso_test = X_test[np.where(y_test >= 0.9)[0], :]
ms_test = X_test[np.where(y_test < 0.9)[0]]

#######

yso_weights = np.arange(0.1, 5, 0.1)

modeldir = 'Models/ratio_innate/'
# modeldir = 'Models/ratio_balanced/'

true_false = pickle.load(open(modeldir + 'truefalse.pickle', 'rb'))

true_n = true_false[0]
false_n = true_false[1]
true_p = true_false[2]
false_p = true_false[3]




fig, ax = plt.subplots()
ax.plot(yso_weights, true_p, label = 'True positive')
ax.plot(yso_weights, false_p, label = 'False positive')
ax.plot(yso_weights, false_n, label = 'False negatives')
ax.legend()
plt.show()


precision = true_p / (true_p + false_p)
recall = true_p / (true_p + false_n)
fig, axs = plt.subplots(2,1, sharex = True, gridspec_kw={'height_ratios':[2,1]})
axs[0].plot(yso_weights, precision, label = 'Precision')
axs[0].plot(yso_weights, recall, label = 'Recall')
axs[0].legend()
axs[1].fill_between(yso_weights, true_p / len(yso_test), false_p / len(yso_test), color = 'blue', label = 'True PMS')
axs[1].fill_between(yso_weights, false_p / len(yso_test), color = 'orange', label = 'False PMS')
axs[1].set_xlabel('Model PMS Weight')
axs[1].set_ylabel('Fraction')
axs[1].legend(frameon = False)
plt.show()


fig, ax = plt.subplots()
points = ax.scatter(false_p  , true_p, c = yso_weights)
ax.set_xlabel('False Positive')
ax.set_ylabel('True Positive')
fig.colorbar(points, label = 'Model PMS Weight')
plt.show()

fig, ax = plt.subplots()
points = ax.scatter(false_p / (true_p + false_p) * 100 , true_p , c = yso_weights)
ax.set_xlabel('% Contamination')
ax.set_ylabel('True Positive')
fig.colorbar(points, label = 'Model PMS Weight')
plt.show()

fig, ax = plt.subplots()
points = ax.scatter(false_p / (true_p + false_p) * 100 , true_p / len(yso_test) * 100, c = yso_weights)
ax.set_xlabel('% Contamination')
ax.set_ylabel('% YSOs Recovered')
fig.colorbar(points, label = 'Model PMS Weight')
plt.show()
