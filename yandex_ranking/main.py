# python3.5

"""
 --------- train a LR model for predicting CTR of a query-url pair ---------
 data set: Yandex search log (https://www.kaggle.com/c/yandex-personalized-web-search-challenge#related-papers)
 created by: Jinkai Yu, <jinkaiyu94@gmail.com>
 ----------------------------------------------------------------------------
"""
import numpy as np
import gzip
from yandex_ranking.preprocess import Session
from tqdm import tqdm  # progress bar
# from collections import deque
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

print(__doc__)

NUM_LINES = 1e+5  # about 1e+8 lines in train file in total
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

# step 2: begin train Model 1 (LR) and evaluate.
rounds = 2
seed = 12345
rng = np.random.RandomState(seed)
clf = LogisticRegression(penalty='l2', solver='liblinear', C=1.0)
all_acc, all_loss = [], []
for r in range(rounds):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=rng)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)  # predict class label
    acc = np.mean(y_test == y_pred)
    y_pred_proba = clf.predict_proba(X_test)
    loss = log_loss(y_test, y_pred_proba, labels=[0, 1])
    print("round %d, accuracy: %f, log_loss: %f" % (r+1, acc, loss))
    all_acc.append(acc)
    all_loss.append(loss)
print("average accuracy: %f, average loss: %f" % (np.mean(all_acc), np.mean(all_loss)))

# step 3: 
