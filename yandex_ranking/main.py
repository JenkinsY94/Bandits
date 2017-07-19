# -*- coding: utf-8 -*-
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
 7. Incorporate uncertainty score from Model 2 with TS；set prior of beta(a,b) using 1st stage's ctr.
 8. use FTRL (tensorflow) for baseline.
 -----------------------------------------------------------------------------
"""
import logging
import numpy as np
import gzip
import random
from scipy.sparse import csr_matrix, vstack, hstack
from tqdm import tqdm  # progress bar
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import log_loss, mean_squared_error, precision_score, precision_recall_curve, roc_curve, roc_auc_score
import time
import matplotlib.pyplot as plt
from model import Session
from MAB import TSOverScore, TSOverPosition, TSOverPosScore, TSOverAbsErr
from offline_emulate import check_sess_click, log_k_items


print(__doc__)

NUM_LINES = int(5 * 1e+6)          # maximum number of session to read (about 1e+8 lines in train file in total)
NUM_SESS = int(1e+5)               # maximum number of session to use
TRAIN_DIR = 'input/train.gz'
RAN_SEED = 12345
N_FEATURES = 2 ** 14               # feature dimension for hashing features.
K = 3                              # the position to explore. select from [1,10]. 10 denotes no-explore.
MAX_POS = 10                       # exploration candidate from position [K, MAX_POS]
TS_POLICY = 'abs_err'            # {score, position, pos_score, abs_err}
WEIGHT_SCHEME = 'multinomial'      # {multinomial, propensity, na}
EXP_SETTING = {'num_sess': NUM_SESS, 'feature_dim': N_FEATURES,
               'K': K, 'max_position': MAX_POS, 'E-E policy': TS_POLICY, 'weight': WEIGHT_SCHEME}
logging.basicConfig(filename='exp.log', filemode='a', level=logging.INFO)
logging.info(EXP_SETTING)

# ++++++++++++++++++++++++++++++++++++
# STEP 1: preparing data
# construct sessions.
# ++++++++++++++++++++++++++++++++++++
sessions = []
sess_train, sess_explore, sess_test = [], [], []
with gzip.open(TRAIN_DIR, 'r') as f_train:
    for (idx, line) in tqdm(enumerate(f_train)):
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
random.seed(RAN_SEED)
random.shuffle(sessions)
sessions = sessions[:NUM_SESS]
num_sessions = len(sessions)
print("session example:\n%s" % sessions[-2].to_string())
print("#session to use: %d" % num_sessions)

# split all sessions into 3 parts
# day 1-10 for train, day 11-20 for explore and retrain (for baseline this is also for train),
# day 21-30 for test.
for s in sessions:
    if s.day < 10:
        sess_train.append(s)
    elif s.day < 20:
        sess_explore.append(s)
    else:
        sess_test.append(s)
del sessions
print("proportion of samples in: \ntrain set: %.3f%%, \nexplore set: %.3f%%, \ntest set: %.3f%%\n"
      % (100.0*len(sess_train)/num_sessions, 100.0*len(sess_explore)/num_sessions, 100.0*len(sess_test)/num_sessions))


# construct features from list of sessions.
def construct_feature(sessions, k=10):
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


# evaluate algorithm on test set
def evaluate_algo(clf, test_sess, plot=False, model_name=''):
    num_clicked_sess, num_clicked_item = 0, 0
    num_sess = 0
    y_true, y_pred = [], []
    for test_s in test_sess:
        cur_x, cur_y = construct_feature([test_s], k=10)
        if cur_x is None:
            continue
        cur_y_pred = clf.predict_proba(cur_x)
        cur_y_pred = [i[1] for i in cur_y_pred]  # probability of positive class(clicked).
        y_true += cur_y.tolist()
        y_pred += cur_y_pred
        num_sess += 1
        cur_click_num = check_sess_click(cur_y, cur_y_pred, k=K)
        num_clicked_item += cur_click_num
        if cur_click_num > 0:
            num_clicked_sess += 1
    sess_ctr = 1.0 * num_clicked_sess / num_sess
    item_ctr = 1.0 * num_clicked_item / (K * num_sess)
    auc = roc_auc_score(y_true, y_pred)
    print("Summary of %s:\n CTR(session level): %f\n CTR(item level): %f\n AUC: %f" % (model_name, sess_ctr, item_ctr, auc))
    logging.info("Summary of %s:\n CTR(session level): %f\n CTR(item level): %f\n AUC: %f" % (model_name, sess_ctr, item_ctr, auc))
    if plot:
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve (%s)' % model_name)
        plt.show()
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        plt.figure()
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC (%s)' % model_name)
        plt.show()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# STEP 2: construct features(no-explore) for baseline model LR. train and evaluate.
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
print("="*10 + " baseline model LR " + "="*10)
X_train, y_train = construct_feature(sess_train+sess_explore, k=K)
print("Shape of X_train: %s, shape of y_train: %s" % (X_train.shape, y_train.shape))

clf_base = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, class_weight='balanced', verbose=0)
clf_base.fit(X_train, y_train)

y_pred_tr = clf_base.predict(X_train)  # predict class label
acc = np.mean(y_train == y_pred_tr)
precision = precision_score(y_train, y_pred_tr, labels=[0, 1])
y_pred_proba_tr = clf_base.predict_proba(X_train.toarray())
loss = log_loss(y_train, y_pred_proba_tr, labels=[0, 1])
print("BASELINE(LR) in train-set:\n accuracy: %f, precision: %f, log_loss: %f" % (acc, precision, loss))

evaluate_algo(clf_base, sess_test, plot=False, model_name='BASELINE LR')

del clf_base, X_train, y_train

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# STEP 3: train LR, GBDT; TS explore; evaluate.
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
print("="*10 + " LR + explore using TS " + "="*10)
print("="*10 + "policy: " + TS_POLICY + ", weight scheme: " + WEIGHT_SCHEME + "="*10)
X_train, y_train = construct_feature(sess_train, k=K)
print("Shape of X_train: %s, shape of y_train: %s" % (X_train.shape, y_train.shape))

clf_1 = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, class_weight='balanced', verbose=0)
clf_1.fit(X_train, y_train)

# explore and retrain LR
X_explore = np.zeros((K*len(sess_explore), N_FEATURES))
y_explore = np.zeros(K*len(sess_explore))
sample_weight_exp = []  # re-weight explored items.

suc_times = sum(y_train)               # use empirical average ctr as beta prior param a
fail_times = len(y_train) - suc_times  # use empirical average ctr as beta prior param b
print("\nsetting beta prior beta(a=%f, b=%f)\n" % (suc_times, fail_times))
if TS_POLICY == 'score':
    ts = TSOverScore(a=suc_times, b=fail_times)  # initialize thompson sampling with prior beta(a,b), TSOverScore(a=1, b=1)
elif TS_POLICY == 'position':
    ts = TSOverPosition(k=K, max_pos=MAX_POS, a=suc_times, b=fail_times)
elif TS_POLICY == 'pos_score':
    ts = TSOverPosScore(k=K, max_pos=MAX_POS, a=suc_times, b=fail_times)
elif TS_POLICY == 'abs_err':
    clf_2 = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, loss='ls', verbose=1)  # default setting
    y_pred_clf1 = clf_1.predict_proba(X_train)
    y_pred_clf1 = [item[1] for item in y_pred_clf1]
    abs_err = np.abs(y_train - y_pred_clf1)
    clf_2.fit(X_train, abs_err)
    ts = TSOverAbsErr()
else:
    raise ValueError('cannot resolve TS_POLICY')

valid_exp_sample = 0
for s_exp in sess_explore:
    x_exp, y_exp = construct_feature([s_exp], k=10)
    if x_exp is None:
        continue
    x_exp = x_exp.toarray()
    y_pred_exp = clf_1.predict_proba(x_exp)
    y_pred_exp = [item[1] for item in y_pred_exp]
    if TS_POLICY == 'abs_err':
        abs_err_exp = clf_2.predict(x_exp)
        exp_idx, bucket_idx, weight = ts.thompson_sample(y_pred_exp, abs_err_exp, k=K, max_pos=MAX_POS, weight_type=WEIGHT_SCHEME)
    else:
        exp_idx, bucket_idx, weight = ts.thompson_sample(y_pred_exp, k=K, max_pos=MAX_POS, weight_type=WEIGHT_SCHEME)

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

# ts.show_explore_times()
# ts.show_distribution(bucket_idx=[15, 16, 17])
# ts.show_distribution(bucket_idx=[20, 25, 30])
# ts.show_distribution(bucket_idx=[50, 70, 90])

# retrain with new data collected from explore.
new_X = vstack([X_train, X_explore_ss])
new_y = np.concatenate((y_train, y_explore))
sample_weight_train = [1 for i in range(len(y_train))]
sample_weight = np.asarray(sample_weight_train + sample_weight_exp)

clf_1.fit(new_X, new_y, sample_weight=sample_weight)
evaluate_algo(clf_1, sess_test, plot=False, model_name='LR with TS')
