import numpy as np 
from astropy.io import fits
import matplotlib.pyplot as plt
import pandas as pd


from sklearn.svm import SVC 
# consider using sklearn PIPELINE and sklearn scaling?

orionfile = '/Users/aidanmcbride/Documents/Sagitta-Runaways/master_revised.fits'
f6file = '/Users/aidanmcbride/Documents/Sagitta-Runaways/final6age.fits'
sagittafile = '/Users/aidanmcbride/Documents/Sagitta-Runaways/final6age.fits'

dat = pd.DataFrame(fits.open(f6file))