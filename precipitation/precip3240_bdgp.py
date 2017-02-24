import numpy as np
from scipy.io.matlab import loadmat
import GPy
import climin
from sklearn.cluster import KMeans, MiniBatchKMeans
import time
# Load data
# =========
D = loadmat('data/precipitation3240/2010-100ktest.mat')
X = D["X"]
y = D["y"]
Xtest = D["Xtest"]
ytest = D["ytest"]

ntrain = X.shape[0]
Xtrain = X[:ntrain, :]
ytrain = y[:ntrain]


nkmeans = 30000
Xkm = X[:nkmeans, :]

# Find clusters
ninducing = 1000
batchsize = 1000
niter = 2000
# use less distance for temporal axis than location axis
scale = np.array([1., 1., 1.]) / (Xtrain.max(0) - Xtrain.min(0))
mbk = KMeans(init='k-means++', n_clusters=ninducing, verbose=5)
mbk.fit(Xkm * scale[np.newaxis, :])
centers = mbk.cluster_centers_ / scale[np.newaxis, :]

d = Xtrain.shape[1]
ytm = ytrain.mean()
m = GPy.core.SVGP(Xtrain, ytrain - ytm, centers,
                  GPy.kern.RBF(d, ARD=True) + GPy.kern.White(1),
                  GPy.likelihoods.Gaussian(), batchsize=batchsize)
m.inducing_inputs.fix()
#m.likelihood.fix()
opt = climin.Adadelta(m.optimizer_array, m.stochastic_grad, step_rate=0.001, momentum=0.6)

def callback(i):
    print "Iteration", i['n_iter'], ":  ", m.log_likelihood(), "                ",
    # Stop after 5000 iterations
    if i['n_iter'] > niter :
        return True
    print "\r",
    return False
t = time.time()
info = opt.minimize_until(callback)
t_train = time.time() - t
t = time.time()
y_pred, var_pred = m.predict(Xtest)
t_test = time.time() - t
y_pred = y_pred + ytm

mse = ((y_pred-ytest)**2).mean()
