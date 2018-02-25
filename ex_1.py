from sklearn import datasets
import numpy
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.manifold import Isomap
from sklearn import metrics
from sklearn.metrics import homogeneity_score, completeness_score,\
    v_measure_score, adjusted_rand_score, adjusted_mutual_info_score, silhouette_score
from sklearn import svm


digits = datasets.load_digits()

# słowa kluczowe zbioru danych
print(digits.keys())

# dane
print(digits.data)

# wartości docelowe
print(digits.target)

print(digits.target_names)

print(digits.images)

# Wypisz opis danych `digits`
print(digits.DESCR)

print(numpy.all([digits.images.reshape((1797, 64)) == digits.data]))

images_and_labels = list(zip(digits.images, digits.target))

plt.figure(1)
for index, (image, label) in enumerate(images_and_labels[:8]):
    # Inizjalizacja okien podrzędnych w siatce 2x4 na pozycji i+1
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    # Wyświetl obrazy w każdym oknie podrzędnym, w odcieniach szarości
    plt.imshow(image, cmap=plt.get_cmap('gray_r'), interpolation='nearest')
    # Dodaj tytuł do każdego obrazu
    plt.title('Training: ' + str(label))

# Tworzenie analizy głównych składowych o 2 komponentach
randomized_pca = PCA(n_components=2, svd_solver='randomized')
reduced_data_rpca = randomized_pca.fit_transform(digits.data)
pca = PCA(n_components=2)
reduced_data_pca = pca.fit_transform(digits.data)
print(reduced_data_rpca)
print(reduced_data_pca)

plt.figure(2)
# Tworzenie wykresu punktowego
colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']
for i in range(len(colors)):
    x = reduced_data_rpca[:, 0][digits.target == i]
    y = reduced_data_rpca[:, 1][digits.target == i]
    plt.scatter(x, y, c=colors[i])
plt.legend(digits.target_names, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel('Pierwsza główna składowa')
plt.ylabel('Druga główna składowa')
plt.title("Wykres punktowy PCA")

# PREPROCESSING
data = scale(digits.data)
# test_size domyślnie przyjmuje wartość 0.25 jeśli nie jest zdefiniowany train_size
X_train, X_test, y_train, y_test, images_train, images_test = \
    train_test_split(data, digits.target, digits.images, random_state=42)


n_samples, n_features = X_train.shape

print(n_samples)
print(n_features)
n_digits = len(numpy.unique(y_train))
print(len(y_train))

# GRUPOWANIE (CLUSTERING) ESTYMATOREM K-MEANS - NIEWYSTARCZAJĄCE
clf = KMeans(init='k-means++', n_clusters=10, random_state=42)
clf.fit(X_train)


fig = plt.figure(figsize=(8, 3))
plt.figure(3)
fig.suptitle('Cluster Center Images', fontsize=14, fontweight='bold')
for i in range(10):
    ax = fig.add_subplot(2, 5, i+1)
    ax.imshow(clf.cluster_centers_[i].reshape(8, 8), cmap=plt.get_cmap('binary'))
    plt.axis('off')


y_pred = clf.predict(X_test)
print(y_pred[:100])
print(y_test[:100])

X_iso = Isomap(n_neighbors=10).fit_transform(X_train)
clusters = clf.fit_predict(X_train)
fig1, ax1 = plt.subplots(1, 2, figsize=(8, 4))
plt.figure(4)
fig1.suptitle('Predicted vs Training Labels', fontsize=14, fontweight='bold')
fig1.subplots_adjust(top=0.85)
ax1[0].scatter(X_iso[:, 0], X_iso[:, 1], c=clusters)
ax1[0].set_title('Predicted Training Labels')
ax1[1].scatter(X_iso[:, 0], X_iso[:, 1], c=y_train)
ax1[1].set_title('Actual Training Labels')

X_pca = PCA(n_components=2).fit_transform(X_train)
clusters = clf.fit_predict(X_train)
fig2, ax2 = plt.subplots(1, 2, figsize=(8, 4))
plt.figure(5)
fig2.suptitle('Predicted vs Training Labels', fontsize=14, fontweight='bold')
fig2.subplots_adjust(top=0.85)
ax2[0].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters)
ax2[0].set_title('Predicted Training Labels')
ax2[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_train)
ax2[1].set_title('Actual Training Labels')


# na przekątnej widzimy, ile razy liczba została poprawnie zaklasyfikowana
print(metrics.confusion_matrix(y_test, y_pred))

# inne metryki
# homogenity_score - stopień w jakim zbiory  zawieraja dane należące do jednej klasy
# completness_score - stopień w jakim elementy danej klasy należą do tego samego zbioru
# v_measure_score - średnia harmoniczna pomiędzy homogenity_score i completness_score
# adjusted_rand_score - okresla podobieństwo  pomiędzy dwoma zbiorami
# AMI - porównuje dane w zbiorach
# silhoette_score - okresla, jak bardzo element zbioru jest podobny do swojego zbioru w porównaniu do innych zbiorów

print('% 9s' % 'inertia    homo   compl  v-meas     ARI AMI  silhouette')
print('%i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f' % (clf.inertia_,
      homogeneity_score(y_test, y_pred),
      completeness_score(y_test, y_pred),
      v_measure_score(y_test, y_pred),
      adjusted_rand_score(y_test, y_pred),
      adjusted_mutual_info_score(y_test, y_pred),
      silhouette_score(X_test, y_pred, metric='euclidean')))

# CLUTERING METODĄ SUPPORT VECTOR MACHINES
svc_model = svm.SVC(gamma=0.001, C=100., kernel='linear')
svc_model.fit(X_train, y_train)
print(svc_model.predict(X_train))
print(y_test)

plt.figure(6)
predicted = svc_model.predict(X_test)
images_and_predictions = list(zip(images_test, predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(1, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.get_cmap('gray_r'), interpolation='nearest')
    plt.title('Predicted: ' + str(prediction))

print(metrics.classification_report(y_test, predicted))
print(metrics.confusion_matrix(y_test, predicted))

X_iso = Isomap(n_neighbors=10).fit_transform(X_train)
predicted = svc_model.predict(X_train)
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
fig.subplots_adjust(top=0.85)
ax[0].scatter(X_iso[:, 0], X_iso[:, 1], c=predicted)
ax[0].set_title('Predicted labels')
ax[1].scatter(X_iso[:, 0], X_iso[:, 1], c=y_train)
ax[1].set_title('Actual Labels')
fig.suptitle('Predicted versus actual labels', fontsize=14, fontweight='bold')
plt.show()
