"""
 --------- train a LR model for predicting CTR of a query-url pair ---------
 data set: Yandex search log (https://www.kaggle.com/c/yandex-personalized-web-search-challenge#logs-format)
 created by: Jinkai Yu
 TODO:
 csr_matrix vstack is inefficient.
 Use numpy ndarray during construction and switch to csr_matrix periodically.
 ----------------------------------------------------------------------------
"""
import numpy as np
import gzip
from scipy.sparse import csr_matrix, vstack, hstack
from preprocess import Session
from tqdm import tqdm  # progress bar
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import log_loss, mean_squared_error
import time

print(__doc__)

NUM_LINES = 1 * 1e+4  # about 1e+8 lines in train file in total
TRAIN_DIR = 'input/train.gz'
# SUPPORT_THRESH = 5  # support threshold for category in one-hot feature construction.
N_FEATURES = 2 ** 10  # feature dimension for hashing features.

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

        if idx + 1 == NUM_LINES:
            break
print("session example:\n%s" % sessions[-2].to_string())
print("#session read: %d" % len(sessions))

# construct features from sessions.
X, y = None, None
X_ss, y_ss = None, None  # sparse representation of X, y
cur_val_samples = 0  # record current sample number in X
for s in tqdm(sessions):
    new_x, new_y = s.gen_hash_feature(n_features=N_FEATURES)
    if new_x is None or new_y is None:
        continue
    if X is None:
        X, y = new_x, new_y
    else:
        while cur_val_samples + new_x.shape[0] > X.shape[0]:
            X.resize((2 * X.shape[0], X.shape[1]))
            y.resize(2 * y.shape[0])
        X[cur_val_samples:cur_val_samples + new_x.shape[0]] = new_x
        y[cur_val_samples:cur_val_samples + new_x.shape[0]] = new_y

    cur_val_samples += new_x.shape[0]

    if cur_val_samples >= 1e+3:
        X_ss = vstack([X_ss, csr_matrix(X)])
        y_ss = hstack([y_ss, y])
        cur_val_samples = 0
        X, y = None, None
X = X[:cur_val_samples]
y = y[:cur_val_samples]
X_ss = vstack([X_ss, csr_matrix(X)])
y_ss = hstack([y_ss, y])
y_ss = y_ss.toarray().reshape(y_ss.shape[1])
del X, y
del sessions

print("\nshape of X: %s\nshape of y: %s" % (X_ss.shape, y_ss.shape))

# step 2: begin train Model 1 (LR) and evaluate.
input("start training...")
# rounds = 5
seed = 12345
rng = np.random.RandomState(seed)
clf_1 = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, verbose=1)
clf_2 = GradientBoostingRegressor(n_estimators=20, learning_rate=0.1, max_depth=10, loss='ls', verbose=1)
all_acc, all_loss = [], []

# for r in range(rounds):
X_train, X_test, y_train, y_test = train_test_split(X_ss, y_ss, test_size=0.3, random_state=rng)
clf_1.fit(X_train, y_train)
y_pred = clf_1.predict(X_test.toarray())  # predict class label
acc = np.mean(y_test == y_pred)
y_pred_proba = clf_1.predict_proba(X_test.toarray())
loss = log_loss(y_test, y_pred_proba, labels=[0, 1])
# print("round %d, accuracy: %f, log_loss: %f" % (r + 1, acc, loss))
all_acc.append(acc)
all_loss.append(loss)

# step 3: train Model 2 (GBDT) and evaluate
y_abs_err = np.abs(y_test - y_pred_proba[:, 1])
clf_2.fit(X_test, y_abs_err)
y_pred_2 = clf_2.predict(X_test.toarray())
mse = mean_squared_error(y_test, y_pred_2)

print("\nSummary of M1(LR):\n average accuracy: %f(std: %f)\n average loss: %f(std: %f)" \
      % (np.mean(all_acc), np.std(all_acc), np.mean(all_loss), np.std(all_loss)))
print("\nSummary of M2(GBDT):\n mse: %f" % mse)
# step 3:
