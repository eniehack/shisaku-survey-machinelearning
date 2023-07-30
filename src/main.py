from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, random_state=0,
)

svm = SVC(kernel="rbf", C=10, gamma=0.1).fit(X_train, y_train)
