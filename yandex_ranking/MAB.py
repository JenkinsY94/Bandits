from scipy.stats import beta
import matplotlib.pyplot as plt
import numpy as np


class TSOverScore(object):
    """ Thompson Sampling over scores policy."""
    def __init__(self, bucket_size=100, a=1, b=1):
        """
        :param bucket_size: number of buckets.
        :param a & b: beta prior distribution param. for each arm/bucket,
                      initialize them with beta(1,1) by default, i.e. uniform distribution.
        """
        self.bucket_size = bucket_size
        self.buckets = [{'a': a, 'b': b} for i in range(bucket_size)]
        self.exp_times_each = [0 for i in range(bucket_size)]
        self.exp_times = 0

    def thompson_sample(self, pred_ctr, k, max_pos, weight_type='propensity'):
        """
        :param pred_ctr: list of predicted ctr
        :param k: position to explore.(count from 1)
        :param max_pos: explore candidate's maximum position.(count from 1)
        :param weight_type: 'propensity' or 'multinomial'. otherwise assign unit weight.
        :return: index of item to explore (index count from 0);
                  index of bucket/arm (count from 0);
                  the chosen arm's weight (propensity weight or multinomial weight.)
        """
        assert 10 >= max_pos > k > 1
        active_arms = []
        for i, score in enumerate(pred_ctr[k-1: max_pos]):
            assert 0 <= score <= 1
            temp = dict()
            temp['exp_idx'] = i + k - 1
            bucket_idx = int(score * self.bucket_size) % self.bucket_size
            temp['bucket_idx'] = bucket_idx
            temp['ts_score'] = beta.rvs(self.buckets[bucket_idx]['a'], self.buckets[bucket_idx]['b'])
            active_arms.append(temp)
        arm_chosen = max(active_arms, key=lambda x: x['ts_score'])
        self.exp_times += 1
        self.exp_times_each[arm_chosen['bucket_idx']] += 1
        if weight_type == 'propensity':
            weight = 1.0 * self.exp_times / self.exp_times_each[arm_chosen['bucket_idx']]
        elif weight_type == 'multinomial':
            weight = 1.0 * sum(pred_ctr[k-1: max_pos]) / pred_ctr[arm_chosen['exp_idx']]
        else:
            weight = 1
        return arm_chosen['exp_idx'], arm_chosen['bucket_idx'], weight

    def batch_update(self, bucket_idx, label):
        if label == 1:
            self.buckets[bucket_idx]['a'] += 1
        elif label == 0:
            self.buckets[bucket_idx]['b'] += 1
        else:
            raise ValueError("label must be either 0 or 1.")

    def show_explore_times(self):
        plt.bar(np.arange(self.bucket_size), self.exp_times_each)
        plt.title("explore times of each bucket(TS over score)")
        plt.xlabel("bucket index")
        plt.ylabel("times")
        plt.show()

    def show_distribution(self, bucket_idx):
        """:param bucket_idx: integer or list of integer.
        """
        if type(bucket_idx) == int:
            bucket_idx = [bucket_idx]
        legend = []
        for i in bucket_idx:
            a, b = self.buckets[i]['a'], self.buckets[i]['b']
            x = np.linspace(0, 1, 100)
            line, = plt.plot(x, beta.pdf(x, a, b), label="bucket idx=%d" % i)
            legend.append(line)
        plt.title("beta distribution of buckets")
        plt.legend(handles=legend)
        plt.show()


class TSOverPosition(object):
    """" Thompson Sampling over positions policy."""
    def __init__(self, k=5, max_pos=10, a=1, b=1):
        """
        :param k: start position for explore and replace.(index from 1)
        :param max_pos: explore candidate's maximum position.(index from 1)
        :param a & b: beta prior distribution param
        """
        self.valid_bucket_size = max_pos - k + 1
        self.buckets = [{'a': a, 'b': b} for i in range(max_pos)]
        self.exp_times_each = [0 for i in range(max_pos)]
        self.exp_times = 0

    def thompson_sample(self, pred_ctr, k, max_pos, weight_type='propensity'):
        assert 10 >= max_pos > k > 1
        sample_score = [-1 for i in range(max_pos)]
        for i in range(k-1, max_pos):
            sample_score[i] = beta.rvs(self.buckets[i]['a'], self.buckets[i]['b'])
        exp_idx = sample_score.index(max(sample_score))
        bucket_idx = exp_idx
        self.exp_times += 1
        self.exp_times_each[exp_idx] += 1

        if weight_type == 'propensity':
            weight = 1.0 * self.exp_times / self.exp_times_each[exp_idx]
        elif weight_type == 'multinomial':
            weight = 1.0 * sum(pred_ctr[k-1: max_pos]) / pred_ctr[exp_idx]
        else:
            weight = 1.0

        return exp_idx, bucket_idx, weight

    def batch_update(self, bucket_idx, label):
        if label == 1:
            self.buckets[bucket_idx]['a'] += 1
        elif label == 0:
            self.buckets[bucket_idx]['b'] += 1
        else:
            raise ValueError("label must be either 0 or 1.")


class TSOverPosScore(object):
    """Thompson Sampling over position & score policy."""
    def __init__(self, bucket_size=100, k=5, max_pos=10, a=1, b=1):
        """
        :param bucket_size: size of bucket/arm for each candidate position.
        :param k: start position for explore and replace.(index from 1)
        :param max_pos: explore candidate's maximum position.(index from 1)
        :param a & b: beta prior distribution param
        """
        self.bucket_size = bucket_size
        self.n = max_pos - k + 1
        self.buckets = []
        self.exp_times_each = []
        for i in range(self.n):
            self.buckets.append([{'a': a, 'b': b} for i in range(bucket_size)])
            self.exp_times_each.append([0 for i in range(bucket_size)])
        self.exp_times = 0

    def thompson_sample(self, pred_ctr, k, max_pos, weight_type='propensity'):
        """
        :param pred_ctr: list of predicted ctr
        :param k: position to explore.(count from 1)
        :param max_pos: explore candidate's maximum position.(index from 1)
        :param weight_type: 'propensity' or 'multinomial'. otherwise assign unit weight.
        :return: index of item to explore (index count from 0);
                  index of bucket/arm (count from 0);
                  the chosen arm's weight (propensity weight or multinomial weight.)
        """
        assert 10 >= max_pos > k > 1
        active_arms = []
        for i, score in enumerate(pred_ctr[k-1: max_pos]):
            assert 0 <= score <= 1
            temp = dict()
            temp['exp_idx'] = i + k - 1
            bucket_idx = int(score * self.bucket_size) % self.bucket_size
            temp['bucket_idx'] = (i, bucket_idx)  # 2-level index
            temp['ts_score'] = beta.rvs(self.buckets[i][bucket_idx]['a'], self.buckets[i][bucket_idx]['b'])
            active_arms.append(temp)
        arm_chosen = max(active_arms, key=lambda x: x['ts_score'])
        self.exp_times += 1
        self.exp_times_each[arm_chosen['bucket_idx'][0]]\
                           [arm_chosen['bucket_idx'][1]] += 1

        if weight_type == 'propensity':
            weight = 1.0 * self.exp_times / self.exp_times_each[arm_chosen['bucket_idx'][0]][arm_chosen['bucket_idx'][1]]
        elif weight_type == 'multinomial':
            weight = 1.0 * sum(pred_ctr[k-1: max_pos]) / pred_ctr[arm_chosen['exp_idx']]
        else:
            weight = 1
        return arm_chosen['exp_idx'], arm_chosen['bucket_idx'], weight

    def batch_update(self, bucket_idx, label):
        if label == 1:
            self.buckets[bucket_idx[0]][bucket_idx[1]]['a'] += 1
        elif label == 0:
            self.buckets[bucket_idx[0]][bucket_idx[1]]['b'] += 1
        else:
            raise ValueError("label must be either 0 or 1.")

    def show_explore_times(self):
        plt.title("explore times of each bucket(TS over score)")
        for i in range(self.n):
            plt.subplot(self.n, 1, i+1)
            plt.bar(np.arange(self.bucket_size), self.exp_times_each[i])
            plt.xlabel("bucket index")
            plt.ylabel("times")
        plt.show()


class TSOverAbsErr(object):
    """Thompson sampling over absolute error of baseline's prediction"""
    def __init__(self, bucket_size=100, a=1, b=1):
        self.bucket_size = bucket_size
        self.buckets = [{'a': a, 'b': b} for i in range(bucket_size)]
        self.exp_times_each = [0 for i in range(bucket_size)]
        self.exp_times = 0

    def thompson_sample(self, pred_ctr, pred_abs_err, k, max_pos, weight_type='propensity'):
        """
        :param pred_ctr: list of predicted ctr
        :param pred_abs_err: list of predicted absolute error of predicted ctr.
        :param k: position to explore.(count from 1)
        :param max_pos: explore candidate's maximum position.(count from 1)
        :param weight_type: 'propensity' or 'multinomial'. otherwise assign unit weight.
        :return: index of item to explore (index count from 0);
                  index of bucket/arm (count from 0);
                  the chosen arm's weight (propensity weight or multinomial weight.)
        """
        assert 10 >= max_pos > k > 1
        active_arms = []
        for i, score in enumerate(pred_abs_err[k-1: max_pos]):
            assert 0 <= score <= 1
            temp = dict()
            temp['exp_idx'] = i + k - 1
            bucket_idx = int(score * self.bucket_size) % self.bucket_size
            temp['bucket_idx'] = bucket_idx
            temp['ts_score'] = beta.rvs(self.buckets[bucket_idx]['a'], self.buckets[bucket_idx]['b'])
            active_arms.append(temp)
        arm_chosen = max(active_arms, key=lambda x: x['ts_score'])
        self.exp_times += 1
        self.exp_times_each[arm_chosen['bucket_idx']] += 1
        if weight_type == 'propensity':
            weight = 1.0 * self.exp_times / self.exp_times_each[arm_chosen['bucket_idx']]
        elif weight_type == 'multinomial':
            weight = 1.0 * sum(pred_ctr[k-1: max_pos]) / pred_ctr[arm_chosen['exp_idx']]
        else:
            weight = 1
        return arm_chosen['exp_idx'], arm_chosen['bucket_idx'], weight

    def batch_update(self, bucket_idx, label):
        if label == 1:
            self.buckets[bucket_idx]['a'] += 1
        elif label == 0:
            self.buckets[bucket_idx]['b'] += 1
        else:
            raise ValueError("label must be either 0 or 1.")

