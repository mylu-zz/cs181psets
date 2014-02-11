import numpy as np
import scipy as sp
from scipy import sparse
import pandas as pd
import cPickle as pickle

# Read in data as pandas dataframes
train=pd.read_csv("ratings-train.csv")
users=pd.read_csv("users.csv")
books=pd.read_csv("books.csv")

# First 5 lines of each dataframe
print train.head()
print users.head()
print books.head()

n_feats = 10 #number of features to train
n_users = users.shape[0] #number of users
n_books = len(list(books['ISBN'].unique())) #number of unique books

# Create mapping for books to indices of sparse matrix
book_mappings = dict(zip(list(books['ISBN'].unique()), range(n_books)))
# Add book indices to 'train' DataFrame for later lookup
train['Book_Index'] = [book_mappings[x] for x in train['ISBN']]

# create book feature matrix n_users by n_feats
book_features = pd.DataFrame(np.random.randn(n_feats, n_books), index = range(n_feats), columns = list(books['ISBN'].unique())).as_matrix()
# create user feature matrix n_feats by n_books
user_features = pd.DataFrame(np.random.randn(n_users + 1, n_feats), index = ['None']+list(users['User'].unique()), columns = range(n_feats)).as_matrix()

"""
# Run this section of code to initialize the ratings matrix with training data
# create matrix with users and movie ratings
ratings = sp.sparse.lil_matrix((n_books, n_users+1))
# populate ratings matrix with the book:user ratings
for i in range(200):
	ratings[train['Book_Index'][1000*i:1000*i+999], train['User'][1000*i:1000*i+999]] = train['Rating'][1000*i:1000*i+999]
	print i
ratings[train['Book_Index'][199999], train['User'][199999]] = train['Rating'][199999]

np.save("ratings_matrix.out.npy", ratings)


# read ratings matrix file
with open('ratings_matrix.dat', 'wb') as outfile:
    pickle.dump(ratings, outfile, pickle.HIGHEST_PROTOCOL)

# open ratings matrix from file
with open('ratings_matrix.dat', 'rb') as infile:
    ratings = pickle.load(infile)

"""

THRESHOLD = 1
alpha = 0.0002
beta = 0.0002

# make list of tuples with book index, user index, and rating
rating_datapoints = zip(train['Book_Index'],train['User'],train['Rating'])
converged = False
iteration = 0
step_size = 10

last_user_features = np.matrix.copy(user_features)
last_book_features = np.matrix.copy(book_features)
last_cumulative_error = sys.maxint
while True:
	print iteration
	iteration += 1
	for dp in rating_datapoints:
		error = dp[2] - np.dot(user_features[dp[1],:],book_features[:,dp[0]]) # + beta/2 * (np.sum(np.square(user_features)) + np.sum(np.square(book_features)))
		for feature in range(n_feats):
			user_features[dp[1]][feature] += alpha * (error * book_features[feature,dp[0]] - beta * user_features[dp[1],feature])
			book_features[feature][dp[0]] += alpha * (error * user_features[dp[1],feature] - beta * book_features[feature,dp[0]])
	if iteration % step_size == 0:
		cumulative_error = sum([pow(dp[2] - np.dot(user_features[dp[1],:],book_features[:,dp[0]]),2) + beta/2 * (np.sum(np.square(user_features[dp[1],:])) + np.sum(np.square(book_features[:,dp[0]]))) for dp in rating_datapoints])
		print cumulative_error
		if cumulative_error < THRESHOLD:
			break
		if cumulative_error < last_cumulative_error:
			if step_size > 1:
			    alpha *= 1.25
			last_user_features = np.matrix.copy(user_features)
			last_book_features = np.matrix.copy(book_features)
			last_cumulative_error = cumulative_error
		else:
			alpha *= 0.5
			user_features = last_user_features
			book_features = last_book_features
			step_size = max(1,step_size/2)
	


