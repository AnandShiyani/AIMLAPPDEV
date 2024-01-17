from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets

iris = datasets.load_iris()
print("iris data set loaded")

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size =0.1)
print("Dataset is split into training and testing")
print("Size of training data and its labels:", x_train.shape, y_train.shape)
print("Size of testing data and its labels:", x_test.shape, y_test.shape)

for i in range(len(iris.target_names)):
    print("Label", i , "-", str(iris.target_names[i]))

clf = KNeighborsClassifier(n_neighbors=1)

clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

print("Results of classification using knn with k=1")
for r in range(0, len(x_test)):
    print("Sample:", str(x_test[r]), "Actual label:", str(y_test[r]), "predicted label:", str(y_pred[r]))
print("Classification accuracy:", clf.score(x_test, y_test));

from sklearn.metrics import classification_report, confusion_matrix
print("Confusion Matrix")
print(confusion_matrix(y_test,y_pred))
print("Accuracy metrics")
print(classification_report(y_test,y_pred))

