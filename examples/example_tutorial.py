#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from epoxy import *
import numpy as np
from sklearn.metrics import pairwise
from helpers import *


# ## Baseline: FlyingSquid
# 
# First, we load the L matrices and Y matrices:

# In[ ]:


L_train_orig = np.load('data/L_train.npy')
L_valid_orig = np.load('data/L_valid.npy')
L_test_orig = np.load('data/L_test.npy')

Y_valid = np.load('data/Y_valid.npy')
Y_test = np.load('data/Y_test.npy')


# Let's first see what happens if we train a FlyingSquid model without extending the labeling functions. We'll show numbers on the test set in this notebook, but in our paper we tuned on on the validation set.

# In[3]:


label_model = train_fs_model_spam(L_train_orig)
evaluate_fs_model_spam(label_model, L_test_orig, Y_test)


# As you can see, precision is pretty decent, but recall lags behind.

# ## Improving Recall with Embeddings

# First, let's load up the pre-trained embeddings. We computed these using BERT by generating features for each comment, and taking a global average pool across the tokens in each comment (see `data/df_train.pkl`, etc, for Pandas dataframes of the train/validation/test sets).

# In[4]:


embeddings_train = np.load('data/embeddings_train.npy')
embeddings_valid = np.load('data/embeddings_valid.npy')
embeddings_test = np.load('data/embeddings_test.npy')


# First, we'll do some preprocessing -- compute an index for nearest neighbors. This runs FAISS under the hood, so you can also run it on a GPU if you'd like.

# In[5]:


epoxy_model_train = Epoxy(L_train_orig, embeddings_train, gpu = False)
epoxy_model_train.preprocess(L_train_orig, embeddings_train)

epoxy_model_test = Epoxy(L_train_orig, embeddings_train, gpu = False)
epoxy_model_test.preprocess(L_test_orig, embeddings_test)


# Now, we'll extend the train and test matrices (note that the thresholds were tuned on the validation set -- we are eliding that step for simplicity in this notebook).

# In[6]:


# these thresholds were tuned on the validation set, see paper for details
thresholds = [0.85, 0.85, 0.85, 0.85, 0.81, 0.85, 0.85, 0.88, 0.85, 0.85]

epoxy_model = Epoxy(L_train_orig, embeddings_train)

epoxy_model.preprocess(L_train_orig, embeddings_train)
L_train_extended = epoxy_model.extend(thresholds)

epoxy_model.preprocess(L_test_orig, embeddings_test)
L_test_extended = epoxy_model.extend(thresholds)


# And now let's evaluate training with the extended labeling functions:

# In[7]:


label_model_extended = train_fs_model_spam(L_train_extended)
evaluate_fs_model_spam(label_model_extended, L_test_extended, Y_test)


# As you can see, we have improved recall by more than ten points -- virtually for free!

# ## Training a downstream end model
# 
# And if you'd like, you can still use the label model to generate probabilistic labels for a downstream end model, which gives further improvement (see paper for details):

# In[8]:


training_labels = label_model_extended.predict_proba_marginalized(L_train_extended)

