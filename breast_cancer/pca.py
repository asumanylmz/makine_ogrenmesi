import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.metrics import accuracy_score
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



data = pd.read_csv("breast-cancer-wisconsin.txt")

data=data.replace("?", -9999)
data=data.drop(['Id'], 1)

y=data.Class
x=data.drop(['Class'], 1)


imp = SimpleImputer(missing_values=-9999, strategy="mean")
x = imp.fit_transform(x) #x = β 0 + β 1 x 1 + β 2 x 2 1

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
"""
pca = PCA(n_components=9)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)


print(pca.explained_variance_ratio_,"\n\n\n")  #her bileşen için varyans değerlerni gösteriyor
tahmin = KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='ball_tree', leaf_size=30, p=2,
                                  metric='euclidean', metric_params=None, n_jobs=1) # K en yakın komşu algoritması

tahmin.fit(X_train, y_train)
y_pred = tahmin.predict(X_test)


basari=accuracy_score(y_test,y_pred)
print("\n\n\n%", basari * 100) #tahmininin oranını gösteriyor
print(pca.get_covariance())
"""