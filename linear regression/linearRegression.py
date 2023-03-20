import numpy as np


class LinearRegression:
    def __init__(self,lr=0.001,n_iters=0):
        self.learningRate=lr
        self.numOfIterations=n_iters
        self.weights=None
        self.bias=None
 

 # fit function used for training, X,y are the dataset get from the user
    def fit(self, X, y):
        numOfSamples,numOfFeatures=X.shape
        self.weights=np.zeros(numOfFeatures)#number of weights must be equal to the number of features
        self.bias=0
        #we don't want to predict one sample at a time, we want to tthe prediction for all the samples at the same time
        # so we will have an array having the prediction of each sample    
        # y|pred= wX+b, X|Treanspose=[x1 x2,...xn]^T, wX=[wx1,wx2,...wxn], y|pred=[wx1+b,wx2+b,....wxn+b]
        for i in range(self.numOfIterations):
            yPrediction = np.dot(X,self.weights) + self.bias

            dw= (1/numOfSamples) * np.dot(X.T,(yPrediction-y))
            db= (1/numOfSamples) * np.sum(yPrediction-y)

            self.weights = self.weights - self.learningRate * dw
            self.bias = self.bias - self.learningRate * db
    
    def predict(self,X):
        yPrediction = np.dot(X,self.weights) + self.bias
        return yPrediction



        

        