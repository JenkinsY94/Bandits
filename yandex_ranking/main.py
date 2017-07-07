"""
 ------------------- predicting CTR of a query-url pair ----------------------
 data set: Yandex search log (https://www.kaggle.com/c/yandex-personalized-web-search-challenge#logs-format)
 created by: Jinkai Yu
 NOTE:
 1. csr_matrix vstack is inefficient for incremental construction.
    Use numpy ndarray during construction and switch to csr_matrix periodically.
 2. split data set into 3 parts: 1st for training M1; 2nd for explore based on M1; 3rd for testing new M1.
 Q:
 1. M1 (LR) is not incrementally updated during exploration; only TS's param get updated in real time.
 TODO:
 -- 0. split sessions into 3 parts.
 -- 1. extend model's `gen_hash_feature` to return only top-k items.
 -- 2. add explore module.
 -- 3. X_test returns all features of original session.
 4. set prior of beta(a,b) using 1st stage's ctr.
 5. add weight for ts chosen item.
 6. extend to enable `MAX_POS` > 10, let MAX_POS = len(sess.clicked)
 ----------------------------------------------------------------------------
"""
import numpy as np
import gzip
from scipy.sparse import csr_matrix, vstack, hstack
from tqdm import tqdm  # progress bar
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import log_loss, mean_squared_error
import time
from model import Session
from MAB import TSOverScore
from offline_emulate import check_sess_click, log_k_items

print(__doc__)

NUM_LINES = 1 * 1e+5          # about 1e+8 lines in train file in total
TRAIN_DIR = 'input/train.gz'
N_FEATURES = 2 ** 12          # feature dimension for hashing features.
K = 5                         # the position to explore. select from [1,10]. 10 denotes no-explore.
MAX_POS = 10                  # exploration candidate from position [K, MAX_POS]

# ++++++++++++++++++++++++++++++++++++
# STEP 1: preparing data
# construct sessions.
# ++++++++++++++++++++++++++++++++++++
sessions = []
sess_train, sess_explore, sess_test = [], [], []
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
num_sessions = len(sessions)
print("session example:\n%s" % sessions[-2].to_string())
print("#session read: %d" % num_sessions)

# split all sessions into 3 parts
# day 1-10 for train, day 11-20 for explore and retrain (for baseline this is also for train),
# day 21-30 for test.
for s in sessions:
    if s.day <= 10:
        sess_train.append(s)
    elif s.day <= 20:
        sess_explore.append(s)
    else:
        sess_test.append(s)
del sessions
print("proportion of samples in: \ntrain set: %.3f%%, \nexplore set: %.3f%%, \ntest set: %.3f%%\n"
      % (100.0*len(sess_train)/num_sessions, 100.0*len(sess_explore)/num_sessions, 100.0*len(sess_test)/num_sessions))


# construct features from list of sessions.
def construct_feature(sessions, k=10, max_pos=10):
    X, y = None, None
    X_ss, y_ss = None, None  # sparse representation of X, y
    cur_val_samples = 0  # record current sample number in X
    for s in sessions:
        new_x, new_y = s.gen_k_hash_feature(n_features=N_FEATURES, k=k)
        if new_x is None:
            continue
        if X is None:
            X, y = new_x, new_y
        else:
            while cur_val_samples + new_x.shape[0] > X.shape[0]:
                X.resize((2 * X.shape[0], X.shape[1]), refcheck=False)
                y.resize(2 * y.shape[0], refcheck=False)
            X[cur_val_samples:cur_val_samples + new_x.shape[0]] = new_x
            y[cur_val_samples:cur_val_samples + new_x.shape[0]] = new_y

        cur_val_samples += new_x.shape[0]

        if cur_val_samples >= 1e+3:
            X_ss = vstack([X_ss, csr_matrix(X)])
            y_ss = hstack([y_ss, y])
            cur_val_samples = 0
            X, y = None, None
    if X is None:
        return None, None
    else:
        X = X[:cur_val_samples]
        y = y[:cur_val_samples]
        X_ss = vstack([X_ss, csr_matrix(X)])
        y_ss = hstack([y_ss, y])
        y_ss = y_ss.toarray().reshape(y_ss.shape[1])
        del X, y
        return X_ss, y_ss

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# STEP 2: construct features(no-explore) for baseline model LR, train, and evaluate.
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
print("="*10 + " baseline model LR " + "="*10)
X_train, y_train = construct_feature(sess_train+sess_explore, k=K)
print("Shape of X_train: %s, shape of y_train: %s" % (X_train.shape, y_train.shape))

clf_base = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, verbose=0)
clf_base.fit(X_train, y_train)

y_pred_tr = clf_base.predict(X_train.toarray())  # predict class label
acc = np.mean(y_train == y_pred_tr)
y_pred_proba_tr = clf_base.predict_proba(X_train.toarray())
loss = log_loss(y_train, y_pred_proba_tr, labels=[0, 1])
print("BASELINE(LR) in train-set:\n accuracy: %f, log_loss: %f" % (acc, loss))

num_clicked_sess, num_clicked_item = 0, 0
num_sess = 0
for s_test in sess_test:
    X_test, y_test = construct_feature([s_test], k=10)
    if X_test is None:
        continue
    y_pred_test = clf_base.predict_proba(X_test.toarray())
    y_pred_test = [i[1] for i in y_pred_test]  # probability of positive class(clicked).
    num_sess += 1
    cur_click_num = check_sess_click(y_test, y_pred_test, k=K)
    num_clicked_item += cur_click_num
    if cur_click_num > 0:
        num_clicked_sess += 1
base_sess_ctr = 1.0 * num_clicked_sess / num_sess
base_item_ctr = 1.0 * num_clicked_item / (K * num_sess)
print("Summary of BASELINE LR:\n CTR(session level): %f\n CTR(item level): %f" % (base_sess_ctr, base_item_ctr))

del clf_base, X_train, y_train

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# STEP 3: train LR, GBDT; TS explore (with GBDT); evaluate.
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
print("="*10 + " LR + explore using TS " + "="*10)
X_train, y_train = construct_feature(sess_train, k=K)
print("Shape of X_train: %s, shape of y_train: %s" % (X_train.shape, y_train.shape))

clf_1 = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, verbose=0)
# clf_2 = GradientBoostingRegressor(n_estimators=20, learning_rate=0.1, max_depth=10, loss='ls', verbose=1)
clf_1.fit(X_train, y_train)

# explore and retrain LR
X_explore = np.zeros((K*len(sess_explore), N_FEATURES))
y_explore = np.zeros(K*len(sess_explore))
sample_weight_exp = []  # re-weight explored items.
ts = TSOverScore(a=1, b=1)  # initialize thompson sampling with prior beta(a,b)

valid_exp_sample = 0
for s_exp in sess_explore:
    x_exp, y_exp = construct_feature([s_exp], k=10)
    if x_exp is None:
        continue
    x_exp = x_exp.toarray()
    y_pred_exp = clf_1.predict_proba(x_exp)
    y_pred_exp = [item[1] for item in y_pred_exp]

    exp_idx, bucket_idx, weight = ts.thompson_sample(y_pred_exp, k=K, max_pos=MAX_POS, weight_type='NA')

    x_exp, y_exp = log_k_items(x_exp, y_exp, exp_idx, k=K)

    X_explore[valid_exp_sample:valid_exp_sample+K] = x_exp
    y_explore[valid_exp_sample:valid_exp_sample+K] = y_exp
    cur_exp_weight = [1] * (K-1) + [weight]  # set the explored item's weight
    sample_weight_exp += cur_exp_weight
    valid_exp_sample += K

    ts.batch_update(bucket_idx, y_exp[K-1])  # real time update, no delay.

X_explore = X_explore[:valid_exp_sample]
X_explore_ss = csr_matrix(X_explore)
del X_explore
y_explore = y_explore[:valid_exp_sample]

# retrain with new data collected from explore.
new_X = vstack([X_train, X_explore_ss])
new_y = np.concatenate((y_train, y_explore))
sample_weight_train = [1 for i in range(len(y_train))]
sample_weight = np.asarray(sample_weight_train + sample_weight_exp)

clf_1.fit(new_X, new_y, sample_weight=sample_weight)

# evaluate it on test set
num_clicked_sess, num_clicked_item = 0, 0
num_sess = 0
for s_test in sess_test:
    X_test, y_test = construct_feature([s_test], k=10)
    if X_test is None:
        continue
    y_pred_test = clf_1.predict_proba(X_test.toarray())
    y_pred_test = [item[1] for item in y_pred_test]  # probability of positive class(clicked).
    num_sess += 1
    cur_click_num = check_sess_click(y_test, y_pred_test, k=K)
    num_clicked_item += cur_click_num
    if cur_click_num > 0:
        num_clicked_sess += 1
base_sess_ctr = 1.0 * num_clicked_sess / num_sess
base_item_ctr = 1.0 * num_clicked_item / (K * num_sess)
print("Summary of LR + TS_score:\n CTR(session level): %f\n CTR(item level): %f" % (base_sess_ctr, base_item_ctr))
