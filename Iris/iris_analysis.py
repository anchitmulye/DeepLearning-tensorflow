from sklearn.datasets import load_iris
import matplotlib.pylab as plt
iris = load_iris()

from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

size = []
kn = []
naive = []
svm = []

for i in [0.60,0.625,0.65,0.675,0.70,0.725,0.75,0.775,0.80]:
    size.append(i)

    X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'],
    random_state=0,train_size = i)
                  
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    kn.append(knn.score(X_test, y_test))
    
    gaussian = GaussianNB().fit(X_train, y_train)
    naive.append(gaussian.score(X_test, y_test))
    
    svc = SVC(kernel='rbf', C=10, gamma=0.1)
    svc.fit(X_train, y_train)
    svm.append(svc.score(X_test, y_test))
    
    
plt.plot(size,naive,'--ro')
plt.plot(size,kn,'--go')
plt.plot(size,svm,'--bo')
plt.xlabel('Train Size')
plt.ylabel('Accuracy')
plt.legend()

