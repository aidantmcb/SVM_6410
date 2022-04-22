# SVM6410Final
_Final project for ASTR6410, Spring 2022_

Exploring SVM models for classification of young stellar objects (YSOs). 

Model inputs: absolute magnitudes of Gaia and 2MASS photometry.
Model outputs: True/False flag of whether each star is a YSO.

----------

A dataset can be generated using `svm_datasets.py`. Input features (variable names with `X`) look like

| ABS(G) | ABS(BP) | ABS(RP) | ABS(J) | ABS(H) | ABS(K) | 
| ------ | ------- | ------- | ------ | -------| -------| 
|   ...  |   ...   |   ...   |   ...  |  ...   |   ...  |
|   ...  |   ...   |   ...   |   ...  |  ...   |   ...  |
|   ...  |   ...   |   ...   |   ...  |  ...   |   ...  |

Target features (variable names with `y`) look like

|  PMS  |
| ----- |
| True  |
| False |
| ...  |

For model training, the default is to return subsets X_train, y_train, and X_test, y_test. The model is trained on X_train, y_train and evaluated on X_test, y_test.

-----

SVM models were first trained in `svm_main_w.py`, which did a parameter search across class weights. Summary plots for this were made in `svm_summaryplots_w.py`.

---


Next, using a few class weight slices from the previous step, new SVM models were trained in `svm_main_C.py`, which did a parameter search across the regularization parameter $C$ (controlling how much support vectors can intrude into the hyperplane margins). Summary plots for this were made in `svm_summaryplots_w.py`.

