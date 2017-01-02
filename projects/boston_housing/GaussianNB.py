import numpy as np
from sklearn.naive_bayes import GaussianNB

def train(X, Y):
    clf = GaussianNB()
    clf.fit(X,Y)
    print(clf.predict([[-0.8, -1]]))

def main():
    X = np.array([[-1,-1], [-2, -1], [-3, -2], [1, 1], [2, 1],[3, 2]])
    Y = np.array([1, 1, 1, 2, 2, 2])
    train(X, Y)


if __name__ == "__main__":
    main()
