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

X_test = X[test, :]
y_test = y[test]

yso_test = X_test[np.where(y_test >= 0.9)[0], :]
ms_test = X_test[np.where(y_test < 0.9)[0]]

pms_inds, ms_inds = (np.where(y_train==1)[0], np.where(y_train==0)[0])
len1 = len(pms_inds)
len2 = len(ms_inds)
ind = np.concatenate([np.random.choice(pms_inds,size = len1), np.random.choice(ms_inds, size = len2)])

sampsize = 10000
samp = np.random.choice(len(ind), size = sampsize)
sample_indices = ind[samp]

yso_weights = np.arange(0.1, 5, 0.1)

truepos_yso = np.zeros(len(yso_weights))
falsepos_yso = np.zeros(len(yso_weights))
truepos_old = np.zeros(len(yso_weights))
falsepos_old = np.zeros(len(yso_weights))

# from matplotlib.backends.backend_pdf import PdfPages
# # with PdfPages('Plots/Models_Confusion.pdf') as pdf:
# with np.errstate(divide = 'ignore'):
#     for i in range(len(yso_weights)):
#         weight = yso_weights[i]
#         print('Weight:', weight)
#         # model = Pipeline([('scaler', StandardScaler()), ('SVC', SVC(kernel = 'rbf', probability = False,
#         #             class_weight ={0: 1, 1: weight}))])
#         # model.fit(X_train[sample_indices,:], y_train[sample_indices])

#         fname = 'Models/model_w{}.pickle'.format(str(round(weight, 1)))
#         # pickle.dump(model, open(fname, 'wb'))

#         model = pickle.load(open(fname, 'rb'))


#         y_predict = model.predict(X_test)

#         yso = X_test[np.where(y_predict >= 0.9)[0], :]
#         ms = X_test[np.where(y_predict < 0.9)[0]]

#         yso_real = X_test[np.where(y_test == 1)[0], :]
#         ms_real = X_test[np.where(y_test == 0)[0], :]
#         print('PREDICTED YSO FRAC:', len(yso)/len(ms))
#         print('REAL YSO FRAC:', len(yso_real) / len(ms_real))

#         # fig, ax = plt.subplots(figsize = (8,6))
#         # ax.scatter(ms[:, 1] - ms[:, 2], ms[:,0], c = 'grey', alpha = 0.5)
#         # ax.scatter(yso[:, 1] - yso[:, 2], yso[:,0], c = 'k', alpha = 0.5 )
#         # ymin, ymax = ax.get_ylim()
#         # ax.set_ylim(ymax, ymin)
#         # plt.show()

#         # fig, ax = plt.subplots(figsize = (8,6))

#         # ax.scatter(ms_real[:, 1] - ms_real[:, 2], ms_real[:,0], c = 'grey', alpha = 0.5)
#         # ax.scatter(yso_real[:, 1] - yso_real[:, 2], yso_real[:,0], c = 'k', alpha = 0.5 )
#         # ymin, ymax = ax.get_ylim()
#         # ax.set_ylim(ymax, ymin)
#         # plt.show()


#         cm = confusion_matrix(y_test, y_predict)#, normalize = 'pred')
#         # disp = ConfusionMatrixDisplay(cm, display_labels = ['NOT PMS', 'PMS'])
#         # disp.plot()
#         # ax = disp.ax_
#         # fig = disp.figure_

#         # ax.set_title('Weight: ' + str(weight))
#         # pdf.savefig(fig)
#         # plt.close()

#         truepos_old[i] = cm[0, 0]
#         falsepos_old[i] = cm[1, 0]
#         falsepos_yso[i] = cm[0, 1]
#         truepos_yso[i] = cm[1, 1]

# true_false = np.array([truepos_old, falsepos_old, truepos_yso, falsepos_yso])
# pickle.dump(true_false, open('truefalse.pickle', 'wb'))

true_false = pickle.load(open('truefalse.pickle', 'rb'))

true_n = true_false[0]
false_n = true_false[1]
false_p = true_false[3]
true_p = true_false[2]




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
axs[1].set_ylabel('N Stars / Total PMS sample')
axs[1].legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(precision, recall)
plt.show()

fig, ax = plt.subplots()
points = ax.scatter(false_p  , true_p, c = yso_weights)
ax.set_xlabel('False Positive')
ax.set_ylabel('True Positive')
fig.colorbar(points, label = 'Model PMS Weight')
plt.show()

fig, ax = plt.subplots()
points = ax.scatter(false_p / true_p , true_p , c = yso_weights)
ax.set_xlabel('% Contamination')
ax.set_ylabel('True Positive')
fig.colorbar(points, label = 'Model PMS Weight')
plt.show()

