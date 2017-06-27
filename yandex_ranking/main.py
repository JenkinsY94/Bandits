# python3.5

"""
 --------- train a LR model for predicting CTR of a query-url pair ---------
 data set: Yandex search log (https://www.kaggle.com/c/yandex-personalized-web-search-challenge#related-papers)
 created by: Jinkai Yu, <jinkaiyu94@gmail.com>
 ----------------------------------------------------------------------------
"""
import numpy as np
import gzip
from preprocess import Session
from tqdm import tqdm  # progress bar
# from collections import deque
from sklearn.linear_model import LogisticRegression

print(__doc__)

NUM_LINES = 1e+4  # about 1e+8 lines in train file in total
TRAIN_DIR = 'input/train.gz'

# step 1: preparing data
# construct sessions.
sessions = []
with gzip.open(TRAIN_DIR, 'r') as f_train:
    for (idx, line) in enumerate(f_train):
        line = line.decode('utf-8')  # decode byte to string
        line = line.strip().split('\t')

        if line[1] == 'M':  # meta
            sessions.append(Session(line))
        elif line[2] == 'Q':  # query
            sessions[-1].add_record(line)
        elif line[2] == 'C':  # click
            sessions[-1].add_click(line)
        else:
            raise ValueError("cannot resolve this line: \n%s" % line)

        if idx+1 == NUM_LINES:
            break
print("session example:\n%s" % sessions[-2].to_string())
print("#session read: %d" % len(sessions))
print("#unmatched urls: %d \n(possible reason: the clicked url's position is larger than 10 ?)"
      % Session.not_match_cnt)

# construct features from sessions.
# consider using collections.queue for storing sessions, to improve efficiency.
X, y = None, None
categories = Session.gen_category(sessions, sup_thresh=2)
for s in tqdm(sessions):
    new_x, new_y = s.gen_feature(categories)
    if new_x is None or new_y is None:
        continue
    assert new_x.shape[0] == new_y.shape[0]
    if X is None:
        X, y = new_x, new_y
    else:
        X = np.concatenate((X, new_x), axis=0)
        y = np.concatenate((y, new_y), axis=0)
del sessions
print("\nshape of X: %s\nshape of y: %s" % (X.shape, y.shape))

# step 2: begin training

# step 3: evaluate performance on test set.
