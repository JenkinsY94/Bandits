# python3.5

"""
 --------- train a LR model for predicting CTR of a query-url pair ---------
 data set: Yandex search log (https://www.kaggle.com/c/yandex-personalized-web-search-challenge#related-papers)
 created by: Jinkai Yu, <jinkaiyu94@gmail.com>

"""
import numpy as np
import gzip
from preprocess import Session
# from collections import deque
from sklearn.linear_model import LogisticRegression

NUM_LINES = 1e+6  # about 1e+8 lines in train file in total
TRAIN_DIR = 'input/train.gz'

# step 1: preparing data
# construct sessions.
sessions = []
with gzip.open(TRAIN_DIR, 'r') as f_train:
    for (idx, line) in enumerate(f_train):
        line = line.decode('utf-8') # decode byte to string
        line = line.strip().split('\t')

        if line[1] == 'M':
            sessions.append(Session(line))
        elif line[2] == 'Q':
            sessions[-1].add_record(line)
        elif line[2] == 'C':
            sessions[-1].add_click(line)
        else:
            raise ValueError("cannot resolve this line: \n%s" % line)

        if idx+1 == NUM_LINES:
            print(sessions[-1].to_string())
            break
print("Number of session: %d" % len(sessions))
print("Number of unmatched urls: %d \n(possible reason: the clicked url's position is larger than 10.)"
      % Session.notMatchCnt)

# construct features from sessions.
# consider using collections.queue for storing sessions, to improve efficiency.
X, y = None, None
for s in sessions:
    new_x, new_y = s.gen_feature()
    X = np.concatenate((X, new_x), axis=0)
    y = np.concatenate((y, new_y), axis=0)
del sessions

# step 2: begin training

# step 3: evaluate performance on test set.
