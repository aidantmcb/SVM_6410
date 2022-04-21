import numpy as np 
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.table import Table
import pickle


from sklearn.svm import SVC 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

orionfile = '/Users/aidanmcbride/Documents/Sagitta-Runaways/master_revised.fits'
f6file = '/Users/aidanmcbride/Documents/Sagitta-Runaways/final6age.fits'
sagittafile = '/Users/aidanmcbride/Documents/Sagitta-Runaways/final6age.fits'

data = fits.open(f6file)[1].data

G = data['G'] - 5 * (np.log10(1000/data['PARALLAX']) - 1)
BP = data['BP'] - 5 * (np.log10(1000/data['PARALLAX']) - 1)
RP = data['RP'] - 5 * (np.log10(1000/data['PARALLAX']) - 1)
J = data['J'] - 5 * (np.log10(1000/data['PARALLAX']) - 1)
H = data['H'] - 5 * (np.log10(1000/data['PARALLAX']) - 1)
K = data['k'] - 5 * (np.log10(1000/data['PARALLAX']) - 1)
X = np.array([G, BP, RP, J, H, K]).T.astype(np.float64)

# X = np.array([data['G'], data['BP'], data['RP'], data['J'], data['H'], data['K']]).T.astype(np.float64)
mask = np.where(np.all(np.isnan(X) == False, axis = 1))[0]

data = data[mask]
X = X[mask, :]
y = data['pms']

train = np.where(data['train_set'])[0]
test = np.where(data['test_set'])[0]

X_train = X[train, :]
y_train = y[train]

pms_inds, ms_inds = (np.where(y_train==1)[0], np.where(y_train==0)[0])
len1 = len(pms_inds)
len2 = len(ms_inds)
ind = np.concatenate([np.random.choice(pms_inds,size = len1), np.random.choice(ms_inds, size = len2)])

sampsize = 10000
samp = np.random.choice(len(ind), size = sampsize)
sample_indices = ind[samp]

model = Pipeline([('scaler', StandardScaler()), ('SVC', SVC(kernel = 'rbf', probability = False,
            class_weight ={0: 1, 1: 2}))])
model.fit(X_train[sample_indices,:], y_train[sample_indices])

fname = 'model.pickle'
pickle.dump(model, open(fname, 'wb'))

# model = pickle.load(open(fname, 'rb'))

X_test = X[test, :]
y_test = y[test]

y_predict = model.predict(X_test)

yso = X_test[np.where(y_predict >= 0.9)[0], :]
ms = X_test[np.where(y_predict < 0.9)[0]]

yso_real = X_test[np.where(y_test == 1)[0], :]
ms_real = X_test[np.where(y_test == 0)[0], :]
print('PREDICTED YSO FRAC:', len(yso)/len(ms))
print('REAL YSO FRAC:', len(yso_real) / len(ms_real))

fig, ax = plt.subplots(figsize = (8,6))
ax.scatter(ms[:, 1] - ms[:, 2], ms[:,0], c = 'grey', alpha = 0.5)
ax.scatter(yso[:, 1] - yso[:, 2], yso[:,0], c = 'k', alpha = 0.5 )
ymin, ymax = ax.get_ylim()
ax.set_ylim(ymax, ymin)
plt.show()

fig, ax = plt.subplots(figsize = (8,6))

ax.scatter(ms_real[:, 1] - ms_real[:, 2], ms_real[:,0], c = 'grey', alpha = 0.5)
ax.scatter(yso_real[:, 1] - yso_real[:, 2], yso_real[:,0], c = 'k', alpha = 0.5 )
ymin, ymax = ax.get_ylim()
ax.set_ylim(ymax, ymin)
plt.show()

cm = confusion_matrix(y_test, y_predict, normalize = 'pred')
disp = ConfusionMatrixDisplay(cm, display_labels = model.classes_)
disp.plot()
plt.show()
