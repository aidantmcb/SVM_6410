#################################################################################################
########################################## svm_utils.py ##########################################
# This is a secondary file for this project. 
# Purpose: provide a few miscellaneous functions (e.g. matplotlib formatting)

import numpy as np 
import matplotlib.pyplot as plt


def plotFormatting():
    import matplotlib
    matplotlib.rcParams.update({'xtick.labelsize':18,
                            'ytick.labelsize':18,
                            'axes.titlesize':18,
                            'axes.labelsize':18,
                            'legend.fontsize':11,
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

