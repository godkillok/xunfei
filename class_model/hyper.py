from hpsklearn import HyperoptEstimator, any_sparse_classifier, tfidf
from sklearn.datasets import fetch_20newsgroups
from sklearn import metrics
from hyperopt import tpe
import numpy as np

# Download the data and split into training and test sets

train = fetch_20newsgroups( subset='train' )
test = fetch_20newsgroups( subset='test' )
X_train = train.data
y_train = train.target
X_test = test.data
y_test = test.target

estim = HyperoptEstimator( classifier=any_sparse_classifier('clf'),
                            preprocessing=[tfidf('tfidf')],
                            algo=tpe.suggest, trial_timeout=300)

estim.fit( X_train, y_train )

print( estim.score( X_test, y_test ) )
# <<show score here>>
print( estim.best_model() )
# <<show model here>>