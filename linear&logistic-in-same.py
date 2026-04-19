import numpy as np

class baseRegression:
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
            y_predicted=np.dot(X,self.w)+self.b
            dw=(1/n_samples)*np.dot(X.T,(y_predicted-y))
            db=(1/n_samples)*np.sum(y_predicted-y)

            self.w-=self.learning_rate*dw
            self.b-=self.learning_rate*db

    

    def predict(self,X):
        return self._approximation(X, self.w, self.b)

    def _approximation(self, X, w, b):
        raise NotImplementedError()
        
    