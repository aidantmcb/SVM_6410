import numpy as np 
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.table import Table
import pickle

from sklearn.svm import SVC 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plotFormatting():
    import matplotlib
    matplotlib.rcParams.update({'xtick.labelsize':18,
                            'ytick.labelsize':18,
                            'axes.titlesize':18,
                            'axes.labelsize':18,
                            'font.size':18,
                            'xtick.top':True,
                            'xtick.minor.visible':True,
                            'ytick.minor.visible':True,
                            'xtick.major.size':3,
                            'xtick.minor.size':1.5,
                            'ytick.major.size':3,
                            'ytick.minor.size':1.5,
                            'ytick.right':True,
                            'xtick.direction':'in',
                            'ytick.direction':'in',
                            'font.family':'serif'})

def hr(G, BP, RP, c = None, label = None, ax = None):
    if ax == None:
        fig, ax = plt.subplots(figsize = (8,6))
    ax.scatter(BP - RP, G, c = c, label = label)
