# import libraries
import numpy as np
import scipy as sp
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd

# Read in data as pandas dataframes
train=pd.read_csv("ratings-train.csv")
users=pd.read_csv("users.csv")
books=pd.read_csv("books.csv")

# First 5 lines of each dataframe
print train.head()
print users.head()
print books.head()

# Draws a sample from the conditional posterior distribution of the book-specific biases
def book_feat_draw(X_u, Y_u, var, book_precision):
    Q = np.dot((1/var)*(X_u.T) ,(X_u)) + book_precision
    return np.random.multivariate_normal(np.dot(np.dot(np.linalg.inv(Q),(1/var)*X_u.T), Y_u), np.linalg.inv(Q))

# Draws a sample from the conditional posterior distribution of the user-specific biases
def user_feat_draw(X_m, Y_m, var, user_precision):
    Q = np.dot((1/var)*(X_m.T) ,(X_m)) + user_precision
    return np.random.multivariate_normal(np.dot(np.dot(np.linalg.inv(Q), (1/var)*X_m.T), Y_m), np.linalg.inv(Q))

def gibbs_sampler(data, n_feats, n_post_samples, book_precision_diag, user_precision_diag, progress=True):
    data = data.copy()
    N = data.shape[0]
    #Create indices that allow us to map users and restaurants to rows
    #in parameter vectors.
    uusers, uidx = np.unique(data.User, return_inverse=True)
    uitems, midx = np.unique(data.ISBN, return_inverse=True)
    nusers = uusers.size
    nitems = uitems.size
    #Add numerical indices to dataframe.
    data["uidx"] = uidx
    data["midx"] = midx
    #Group observations by user and by business.
    ugroups = data.groupby("uidx")
    mgroups = data.groupby("midx")
    all_avg = data.Rating.mean()
    u_avg = ugroups.Rating.mean()
    m_avg = mgroups.Rating.mean()
    #Initialize parameters and set up data structures for
    #holding draws.
    #Overall mean
    mu = all_avg
    mu_draws = np.zeros(n_post_samples)
    #Residual variance
    var = 0.5
    var_draws = np.zeros(n_post_samples)
    #Matrix of user-specific bias and n_feats latent factors.
    theta = np.zeros([nusers, n_feats+1])
    theta[:,0] = u_avg-all_avg
    theta_draws = np.zeros([nusers, n_feats+1, n_post_samples])
    #Matrix of item-specific bias and n_feats latent factors.
    gamma = np.zeros([nitems, n_feats+1])
    gamma[:,0] = m_avg-all_avg
    gamma_draws = np.zeros([nitems, n_feats+1, n_post_samples])
    #Matrix for holding the expected number of rating
    #for each observation at each draw from the posterior.
    EY_draws = np.zeros([data.shape[0], n_post_samples])
    #Inverse covariance matrices from the prior on each book_feat
    #and gamma_b. These are diagonal, like Ridge regression.
    book_precision = np.eye(n_feats+1)*book_precision_diag
    user_precision = np.eye(n_feats+1)*user_precision_diag
    #Main sampler code
    for i in range(n_post_samples):
        if i%1==0 and progress:
            print i
        #The entire regression equation except for the overall mean.
        nomu = np.sum(theta[data.uidx,1:]*gamma[data.midx,1:], axis=1) +\
                  theta[data.uidx,0] + gamma[data.midx,0]
        #Compute the expectation of each observation given the current
        #parameter values.
        EY_draws[:,i]=mu+nomu
        #Draw overall mean from a normal distribution
        mu = np.random.normal(np.mean(data.Rating-nomu), np.sqrt(var/N))
        #Draw overall residual variance from a scaled inverse-Chi squared distribution.
        var = np.sum(np.power(data.Rating-nomu-mu,2))/np.random.chisquare(N-2)
        #For each item
        for mi,itemdf in mgroups:
            #Gather relevant observations, and subtract out overall mean and
            #user-specific biases, which we are holding fixed.
            Y_m = itemdf.Rating-mu-theta[itemdf.uidx,0]
            #Build the regression design matrix implied by holding user factors
            #fixed.
            X_m = np.hstack((np.ones([itemdf.shape[0],1]),
                             theta[itemdf.uidx,1:]))
            gamma[mi,:] = user_feat_draw(X_m, Y_m, var, user_precision)
        #For each user
        for ui,userdf in ugroups:
            #Gather relevant observations, and subtract out overall mean and
            #business-specific biases, which we are holding fixed.
            Y_u = userdf.Rating-mu-gamma[userdf.midx,0]
            #Build the regression design matrix implied by holding business factors
            #fixed.
            X_u = np.hstack((np.ones([userdf.shape[0],1]),
                             gamma[userdf.midx,1:]))
            theta[ui,:] = book_feat_draw(X_u, Y_u, var, book_precision)
        #Record draws
        mu_draws[i] = mu
        var_draws[i] = var
        theta_draws[:,:,i] = theta
        gamma_draws[:,:,i] = gamma
    return {"mu": mu_draws, "var": var_draws,
            "theta": theta_draws, "gamma": gamma_draws,
            "EY": EY_draws}

samples = gibbs_sampler(train, 10, 1500, 0.1, 0.1, progress=True)
prediction = [np.mean(sample[1:]) for sample in samples['EY']]

# squared errors
g = np.sum(np.square(np.array(train['Rating']-prediction)))
print g
