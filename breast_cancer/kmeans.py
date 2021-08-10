
import numpy as np
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import pandas as pd
import random
data = pd.read_csv("breast-cancer-wisconsin.txt")


data=data.replace("?", -9999)
data=data.drop(['Id'], 1)

y=data.Class
x=data.drop(['Class'], 1)


imp = SimpleImputer(missing_values=-9999, strategy="mean")
x = imp.fit_transform(x) #x = β 0 + β 1 x 1 + β 2 x 2 1



kmeans = KMeans(n_clusters=9)
kmeans.fit(data)
print(kmeans.cluster_centers_)
print(pd.crosstab(y,kmeans.labels_))



