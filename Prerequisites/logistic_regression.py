from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=42)

lr = LogisticRegression()

lr.fit(X_train, y_train)

predictions = lr.predict(X_test)

accuracy = lr.score(X_test, predictions)
print(accuracy)
