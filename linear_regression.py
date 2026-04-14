#approximation 
#y= ax+b;
#loss function :MSE
#gradient descent


import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as train_test_split
import sklearn.datasets as datasets


class LinearRegression:
    def __init__(self,lr=0.01,n_iters=1000):
        self.learning_rate=lr
        self.n_iters=n_iters
        self.w=None
        self.b=None

    def fit(self,X,y):
        n_samples,n_features=X.shape
        #init parameters
        self.w=np.zeros(n_features)
        self.b=0

        #gradient descent
        for _ in range(self.n_iters):
            y_pred=self.predict(X)
            dw=(1/n_samples)*np.dot(X.T,(y_pred-y))
            db=(1/n_samples)*np.sum(y_pred-y)
            self.w-=self.learning_rate*dw
            self.b-=self.learning_rate*db


    def predict(self,X):
        y_pred=np.dot(X,self.w)+self.b
        return y_pred


def mse(y_true,y_pred):
    return np.mean((y_true-y_pred)**2)


if __name__ == "__main__":
    X,y=datasets.make_regression(n_samples=100,n_features=1,noise=20,random_state=4)
    X_train,X_test,y_train,y_test=train_test_split.train_test_split(X,y,test_size=0.2,random_state=1234)

    fig=plt.figure(figsize=(8,6))
    plt.scatter(X[:,0],y,color="b",marker="o",s=100)
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.title("Linear Regression Data")
    plt.show()

    regressor=LinearRegression(lr=0.01,n_iters=1000)
    regressor.fit(X_train,y_train)
    predicted=regressor.predict(X_test)

    mse_value=mse(y_test,predicted)
    print("MSE:",mse_value)

    y_pred_line=regressor.predict(X)
    cmap=plt.get_cmap("viridis")
    fig=plt.figure(figsize=(8,6))
    m1=plt.scatter(X_train,y_train,color=cmap(0.9),s=10)
    m2=plt.scatter(X_test,y_test,color=cmap(0.5),s=10)
    plt.plot(X,y_pred_line,color="r",label="Regression Line")
    plt.show()