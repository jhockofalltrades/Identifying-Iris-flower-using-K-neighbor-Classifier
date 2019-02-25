from sklearn import datasets #import dataset methods
from sklearn.model_selection import train_test_split #model training
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


iris = datasets.load_iris()


features = iris.data 
lables = iris.target

feature_train, feature_test, labels_train, labels_test = train_test_split(features, lables, test_size=.5)
#In this case, the test size is 50% of the total dataset

my_classifier = KNeighborsClassifier()
my_classifier.fit(feature_train, labels_train)



prediction = my_classifier.predict(feature_test)
 

print(accuracy_score(labels_test,prediction))