from scipy.stats import beta


class TSOverScore(object):
    """ Thompson Sampling over scores policy."""
    def __init__(self, bucket_size=100, a=1, b=1):
        """ for each arm/bucket, initialize them with beta(1,1), i.e. uniform distribution.
        :param bucket_size:
        :param a: beta prior distribution param a.
        :param b: beta prior distribution param b.
        """
        self.bucket_size = bucket_size
        self.buckets = [{'a': a, 'b': b} for i in range(bucket_size)]
        self.exp_times_each = [0 for i in range(bucket_size)]
        self.exp_times = 0

    def thompson_sample(self, pred_ctr, k, max_pos, weight_type='propensity'):
        """
        :param pred_ctr: a list of predicted ctr
        :param k: position to explore.
        :param max_pos: explore candidate's maximum position.
        :param weight_type: 'propensity' or 'multinomial'
        :return: index of item to explore;
                  index of bucket/arm;
                  the chosen arm's weight (propensity weight or multinomial weight.)
        """
        assert 10 >= max_pos > k
        active_arms = []
        for i, score in enumerate(pred_ctr[k: max_pos]):
            assert 0 <= score <= 1
            temp = dict()
            temp['exp_idx'] = i+k
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
            weight = 1.0 * sum(pred_ctr[k: max_pos]) / pred_ctr[arm_chosen['exp_idx']]
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
