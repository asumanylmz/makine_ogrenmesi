import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import metrics
import random
data = pd.read_csv("breast-cancer-wisconsin.txt")

data=data.replace("?",-9999)
"""
data=data.drop(['Id'], 1)

y=data.Class
x=data.drop(['Class'], 1)


imp = SimpleImputer(missing_values=-9999, strategy="mean")
x = imp.fit_transform(x) #x = β 0 + β 1 x 1 + β 2 x 2 1



for i in range (10):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4)

    tahmin = KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='ball_tree', leaf_size=30, p=2,
                                  metric='euclidean', metric_params=None, n_jobs=1) # K en yakın komşu algoritması

    tahmin.fit(X_train, y_train)

    cancer=tahmin.predict(X_test)#cancer=2 ise iyi huylu   4 ise kötü huylu
    basari = metrics.accuracy_score(y_test, cancer)
    print(basari)
"""