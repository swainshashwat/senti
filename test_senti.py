import numpy as np
import pickle
import re
import os
from vectorizer import vect

example = ['I love this movie']

clf = pickle.load(open(os.path.join(
    'movieclassifier',
    'pkl_objects',
     'classifier.pkl'),
      'rb'))

label = {0:'negative', 1:'positive'}


X = vect.transform(example)
print('Prediction: %s' % label[clf.predict(X)[0]])
print('Pred Prob: %.2f%%' % (np.max(clf.predict_proba(X))*100))