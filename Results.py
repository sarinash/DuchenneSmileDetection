import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.plotting import plot_decision_regions
from sklearn import linear_model, svm, metrics
from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
import matplotlib.pyplot as plt
np.seterr(divide='ignore', invalid='ignore')
estimator = SVR(kernel="linear")
selector = RFE(estimator, n_features_to_select=5, step=1)
df = pd.read_excel(r'Results.xlsx')
X = df.drop('Label', axis=1)
y = df['Label']
selector = selector.fit(X, y)
print(selector.support_)
print(selector.ranking_)
X2 = df[['OnsetNetAmplitude', 'OnsetNetAmpDurationRatio', 'ApexDurationN', 'ApexMaximumAmplitude', 'ApexMeanAmplitudeN',
         'ApexMaximumAmplitude', 'ApexMeanAmplitude']]
print(X2)
print(type(X))
X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.4, random_state=109)  # 70% training and 30% test
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred))

