from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import  numpy as np
electric = pd.read_csv("electric.txt")
electric=electric.replace("unstable", 0)
electric=electric.replace("stable",1)
y=electric.stabf
x=electric.drop(['stabf'],1)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
print("tasarım degişkenleri=", X_train.shape[1])
b = ((X_train.shape[1]) * 40) / 100;
print("boyut=",int( b))
boyut=int(b)
""" pca gectık"""
print("----------pca--------------")
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
""" verileri normalize eder"""
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
"""egitim seti kucuk parcalara ayrılır ---> array_slipt 
model egitilir --->partial_fit ile"""
pca = PCA(n_components=boyut)
X_train=pca.fit_transform(X_train)
X_test=pca.fit_transform(X_test)
print("varyans oranı", pca.explained_variance_ratio_)
print ("pcali veri",X_train)
tahmin = KNeighborsClassifier(n_neighbors=5)
tahmin.fit(X_train,y_train)
sonuc=tahmin.predict(X_test)
basari = accuracy_score(y_test,sonuc)
print("basarısı=",basari)
print("-------------------k-means----------------------")
from sklearn.cluster import KMeans
model=KMeans(n_clusters=2)
model.fit(x)
gruplar=np.choose(model.labels_,[1,0]).astype(np.int64)
print(gruplar)
print("basari",accuracy_score(y,gruplar))
print(model.cluster_centers_)
print(" ")
print("------------------svm--------------------")
from sklearn.metrics import  confusion_matrix
from sklearn.svm import SVC
svc = SVC(kernel='gausti')
svc.fit(X_train, y_train)
yson = svc.predict(X_test)
print("confusion matris")#capraz sekilde toplanır 1 ve 4 elemanlar dogru digerleri yanlis
con=confusion_matrix(y_test, yson)
print(con)
topveri=con[0][0]+con[0][1]+con[1][0]+con[1][1]
dogru=con[0][0]+con[1][1]
print(topveri," icinden dogru tahmin",dogru)
yanlis=con[0][1]+con[1][0]
print(topveri,"içinden yanlış tahmin",yanlis)





