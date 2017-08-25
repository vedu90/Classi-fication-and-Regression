import numpy as np
import math
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from numpy.matlib import identity
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD
    N = len(X)
    d = len(X[0])
    #print(y)
    initIndexes = np.linspace(0, N-1, N)
    initIndexes = initIndexes.reshape((N, 1))
    appendedMatrix = np.append(X,initIndexes,axis=1)
    #print("len",len(appendedMatrix[0]))
    y = np.int_(y)
    appendedMatrix = np.append(appendedMatrix,y,axis=1)
    #print("len",len(appendedMatrix[0]))
    appendedMatrix = appendedMatrix[appendedMatrix[:,d+1].argsort()]
    splitMatrix=[]
    
    y.sort(axis=0)
    
    splitMatrix = np.split(appendedMatrix, np.where(y[:-1, 0] != y[1:, 0])[0]+1)
    
    k = len(splitMatrix)
    for i in range(k):
        splitMatrix[i] = splitMatrix[i][splitMatrix[i][:,d].argsort()]
        classLength = len(splitMatrix[i])
        #print("classLength",classLength)
        splitMatrix[i] = np.delete(splitMatrix[i],[d,d+1],1)
        splitMatrix[i] = np.sum(splitMatrix[i],axis=0)
        splitMatrix[i] = splitMatrix[i]/classLength
    #print("Split Matrix length ",splitMatrix[0])
    means = splitMatrix[0]
    means = means.reshape((1, d))
    i = 1
    while i < k:
        splitMatrix[i]= splitMatrix[i].reshape((1, d))
        means = np.append(means,splitMatrix[i],axis=0)
        i+=1
    means = means.transpose()

    covmat = np.cov(X,rowvar=0)
    
    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD

    N = len(X)
    d = len(X[0])
    #print(y)
    initIndexes = np.linspace(0, N-1, N)
    initIndexes = initIndexes.reshape((N, 1))
    appendedMatrix = np.append(X,initIndexes,axis=1)
    #print("len",len(appendedMatrix[0]))
    y = np.int_(y)
    appendedMatrix = np.append(appendedMatrix,y,axis=1)
    #print("len",len(appendedMatrix[0]))
    appendedMatrix = appendedMatrix[appendedMatrix[:,d+1].argsort()]
    splitMatrix=[]
    
    y.sort(axis=0)
    #print(y)
    splitMatrix = np.split(appendedMatrix, np.where(y[:-1, 0] != y[1:, 0])[0]+1)
    #print("Split Matrix length ",splitMatrix[0])
    k = len(splitMatrix)
    covmats = []
    for i in range(k):
        splitMatrix[i] = splitMatrix[i][splitMatrix[i][:,d].argsort()]
        classLength = len(splitMatrix[i])
        #print("classLength",classLength)
        splitMatrix[i] = np.delete(splitMatrix[i],[d,d+1],1)
        covmat= np.cov(splitMatrix[i],rowvar=0)
        #print("covmat ",covmat)
        covmats.append(covmat)
        splitMatrix[i] = np.sum(splitMatrix[i],axis=0)
        splitMatrix[i] = splitMatrix[i]/classLength
    #print("Split Matrix length ",splitMatrix[0])
    means = splitMatrix[0]
    means = means.reshape((1, d))
    i = 1
    while i < k:
        splitMatrix[i]= splitMatrix[i].reshape((1, d))
        means = np.append(means,splitMatrix[i],axis=0)
        i+=1
    means = means.transpose()
     
    #print("Len of covmats ",covmats[0])    
    return means,covmats

    #print(y)
    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD

    N = len(Xtest)
    k = len(means[0])
    #print("ytest",ytest)
    means = means.transpose()
    #print("means[0] : ",means[0])
    diffMatrix = Xtest-means[0]
    
    output = np.dot(np.dot(diffMatrix,inv(covmat)),diffMatrix.transpose())
    output = output.diagonal()
    Mahanalobis = output.reshape((N, 1))
    Mahanalobis = (-0.5)*Mahanalobis
    Mahanalobis = np.exp(Mahanalobis)
    i = 1
    while i < k:
         diffMatrix = Xtest-means[i]
         output = np.dot(np.dot(diffMatrix,inv(covmat)),diffMatrix.transpose())
         output = output.diagonal()
         output = output.reshape((N, 1))
         output = (-0.5)*output
         output = np.exp(output)
         Mahanalobis = np.append(Mahanalobis,output,axis=1)
         i+=1
    #print(Mahanalobis)
    ypred = np.argmax(Mahanalobis,axis=1)
    ypred = ypred.reshape((N, 1))
    ypred+=1
    #print("labels ",np.append(labels,ytest,axis=1))
    acc = 100 * np.mean((ypred == ytest).astype(float))
    #print('\n LDA Validation set Accuracy:' + str(acc) + '%')
    
    return acc,ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD

    N = len(Xtest)
    k = len(means[0])
    #print("ytest",ytest)
    means = means.transpose()
    #print("means[0] : ",means[0])
    diffMatrix = Xtest-means[0]
    covmat = covmats[0]
    output = np.dot(np.dot(diffMatrix,inv(covmat)),diffMatrix.transpose())
    output = output.diagonal()
    Mahanalobis = output.reshape((N, 1))
    Mahanalobis = (-0.5)*Mahanalobis
    Mahanalobis = np.exp(Mahanalobis)
    Mahanalobis = Mahanalobis/np.power(det(covmat),0.5)
    i = 1
    while i < k:
         diffMatrix = Xtest-means[i]
         covmat = covmats[i]
         output = np.dot(np.dot(diffMatrix,inv(covmat)),diffMatrix.transpose())
         output = output.diagonal()
         output = output.reshape((N, 1))
         output = (-0.5)*output
         output = np.exp(output)
         output = output/np.power(det(covmat),0.5)
         Mahanalobis = np.append(Mahanalobis,output,axis=1)
         i+=1
    #print(Mahanalobis)
    ypred = np.argmax(Mahanalobis,axis=1)
    ypred = ypred.reshape((N, 1))
    ypred+=1
    #print("labels ",np.append(labels,ytest,axis=1))
    acc = 100 * np.mean((ypred == ytest).astype(float))
    #print('\nQDA Validation set Accuracy:' + str(acc) + '%')
    
    return acc,ypred

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1 
	
    # IMPLEMENT THIS METHOD
    #print("In learnOLERegression")
    #print("----------------------")
    X_transpose = X.transpose()
    #print("X_transpose : "+ str((X_transpose.shape)))
    firstExp = np.linalg.inv(np.dot(X_transpose, X))
    #print("firstExp : "+ str((firstExp.shape)))
    secondExp = np.dot(X_transpose, y)
    #print("secondExp : "+ str((secondExp.shape)))
    w = np.dot(firstExp, secondExp)
    #print("w : "+ str((w.shape)))
    # IMPLEMENT THIS METHOD                                                   
    return w

    #return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1
    #n = 65
    #print(X.shape)
    #print("In learnRidgeRegression")
    #print("----------------------")
    X_transpose = X.transpose()
    #print("X_transpose : "+ str((X_transpose.shape)))
    firstExp = np.linalg.inv(np.dot(X_transpose, X) + np.multiply(lambd, identity(X.shape[1])))
    #print("firstExp : "+ str((firstExp.shape)))
    secondExp = np.dot(X_transpose, y)
    #print("secondExp : "+ str((secondExp.shape)))
    w = np.dot(firstExp, secondExp)
    #print("w : "+ str((w.shape)))
    # IMPLEMENT THIS METHOD                                                   
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse
    
    # IMPLEMENT THIS METHOD

    sum = 0
    for i in range(len(Xtest)) :
        sum = sum + (ytest[i] - np.dot(w.transpose(), Xtest[i].transpose()))**2
    #print("The sum is :" + str(sum))
    mse = sum / len(Xtest)
    #print("The shape of mse is :" + str(mse.shape))
    # IMPLEMENT THIS METHOD
    return mse
    #return
    #return mse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD
    w = w.reshape(len(w), 1)
    prod = np.dot(X,w)
    N = len(X)
    diff = y-prod
    squareLoss = np.dot(diff.transpose(),diff)
    squareLoss/=2

    regularizedLoss = np.dot(w.transpose(),w)
    regularizedLoss*=(lambd*0.5)
    
    error = squareLoss+regularizedLoss

    error_grad1 = -np.dot(y.transpose(),X)+np.dot(w.transpose(),np.dot(X.transpose(),X))
    error_grad2 = lambd*(w.transpose())
    error_grad = error_grad1+error_grad2
    error_grad = np.array(error_grad).flatten()
    
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xd - (N x (p+1)) 
	
    # IMPLEMENT THIS METHOD
    N = x.shape[0]
    temp = np.empty([N,p+1], dtype = float)
    for i in range (N):
        for j in range (p+1):
            temp[i,j] = math.pow(x[i],j)
    #print(temp)
    return temp


# Main script

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('QDA')

plt.show()
# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('MSE without intercept '+str(mle))
print('MSE with intercept '+str(mle_i))
plt.title("Weight values in Linear Regression")
x = np.linspace(0, 65, 65)
plt.plot(x, w_i)
plt.show()

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
min3 = sys.maxsize
minLamda = 0
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    if(mses3[i] < min3) :
        min3 = mses3[i]
        minLamda = lambd
        w_opt = w_l 
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.show()
plt.title("Weight values in Ridge Regression")
x = np.linspace(0, 65, 65)
plt.plot(x, w_opt)
plt.show()
# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 30}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()


# Problem 5
pmax = 7
minLambda = np.argmin(mses3)
#print(minLambda)
lambda_opt = lambdas[minLambda] # REPLACE THIS WITH lambda_opt estimated from Problem 3
print("optimal lambda value = " + str(lambda_opt))
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()
