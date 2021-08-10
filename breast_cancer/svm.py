import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

url = "iris.data"

# Assign colum names to the dataset
colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']


# Read dataset to pandas dataframe
irisdata = pd.read_csv(url, names=colnames)

X = irisdata.drop('Class', axis=1)
y = irisdata['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)

#iki sınıf varsa linear svm kullanabiliriz ama 2'den fazla sınıf olduğunda kernel svm kullanacağız  . svm verileri ayırmak için çizgi çizer
svclassifier = SVC(kernel='poly', degree=3)
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


