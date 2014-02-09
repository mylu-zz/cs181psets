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

n_feats = 50 #number of features to train
n_users = users.shape[0] #number of users
n_books = books.shape[0] #number of books

# Create mapping for books to indices of sparse matrix
book_mappings = dict(zip(list(books['ISBN'].unique()), range(n_books)))
# Add book indices to 'train' DataFrame for later lookup
train['Book_Index'] = [book_mappings[x] for x in train['ISBN']]

# create book feature matrix
book_features = pd.DataFrame(index = list(books['ISBN'].unique()), columns = range(n_feats))
# create user feature matrix
user_features = pd.DataFrame(index = ['None']+list(users['User'].unique()), columns = range(n_feats))

# create matrix with users and movie ratings
ratings = sp.sparse.lil_matrix((n_books, n_users+1))
# populate ratings matrix with the book:user ratings 309
for i in range(200):
  ratings[train['Book_Index'][1000*i:1000*i+999], train['User'][1000*i:1000*i+999]] = train['Rating'][1000*i:1000*i+999]
  print i
#zip(train[book_mappings['ISBN']],train['User']




