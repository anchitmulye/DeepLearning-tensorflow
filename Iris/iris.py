import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

Iris_data = pd.read_csv('Iris.csv')

##Knowing The Data

#print(Iris_data.head())
#
#Iris_data.info()
#
#print(Iris_data.describe())

##Data Visualization

#data = Iris_data.drop('Id', axis = 1)
#
#sns.set(style = "ticks")
#sns.pairplot(data, hue="Species")
#plt.show()


##Training 

X = Iris_data.drop(['Id', 'Species'], axis = 1)
y = Iris_data['Species']


##Zsccore

from scipy import stats
SepalLengthCm = stats.zscore(X['SepalLengthCm'])
SepalWidthCm = stats.zscore(X['SepalWidthCm'])
PetalLengthCm = stats.zscore(X['PetalLengthCm']) 
PetalWidthCm = stats.zscore(X['PetalWidthCm'])


d = {'SepalLengthCm': SepalLengthCm, 'SepalWidthCm': SepalWidthCm, 'PetalLengthCm': PetalLengthCm, 'PetalWidthCm': PetalWidthCm}
X = pd.DataFrame(data=d)

#print(X.shape)
#print(y.shape)


##MinMax
#
#from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler()
#scaler.fit(X)
##print(df)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#print(X_train)
#print(X_test)

#print(X_train['SepalLengthCm'])

#print(X_train)
#print(df)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
score_knn = knn.score(X_test, y_test)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
score_logreg = logreg.score(X_test, y_test)

from sklearn.svm import SVC, LinearSVC
lsvc = LinearSVC()
svc = SVC(kernel='rbf', C=10, gamma=0.1)
svc.fit(X_train, y_train)
lsvc.fit(X_train, y_train)
score_svc = svc.score(X_test, y_test)
score_lsvc = lsvc.score(X_test, y_test)

from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
score_forest = random_forest.score(X_test, y_test)

from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
score_tree = decision_tree.score(X_test, y_test)

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(random_state=0).fit(X_train, y_train)
score_mlp = mlp.score(X_test, y_test)

from sklearn.naive_bayes import GaussianNB
gaussian = GaussianNB().fit(X_train, y_train)
score_naive = gaussian.score(X_test, y_test)

models = pd.DataFrame({
    'Model': ['KNN', 'Logistic Regression', 'SVC','LinearSVC','Random Forest', 'Decision Tree','MLP','Naive Bayes',],
    'Score': [score_knn, score_logreg, score_svc, score_lsvc, score_forest, score_tree, score_mlp, score_naive]})
print(models.sort_values(by='Score', ascending=False))

