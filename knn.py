
for i in range (10):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4)

    tahmin = KNeighborsClassifier(n_neighbors=5)

    tahmin.fit(X_train,y_train)
    sonuc=tahmin.predict(X_test)

    basari = tahmin.score(X_test, y_test)
    print(basari)


from sklearn.decomposition import IncrementalPCA
n_batches = 100
inc_pca = IncrementalPCA(n_components=150)
"""egitim seti kucuk parcalara ayrılır ---> array_slipt 
model egitilir --->partial_fit ile"""
for X_batch in np.array_split(X_train, n_batches):
        inc_pca.partial_fit(X_test)

X_reduced = inc_pca.transform(X_train)