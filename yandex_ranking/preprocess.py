"""
Session model for pre-processing yandex data set for personalized web search.
Assumes in the same SERP, the 'Q' line exist before the 'C' line.

Issues: ...

TODO:
1. add `position` attribute in Session class.
2. extend the `gen_feature()` method to make use of more categories.
3. include hashing technique to reduce feature dimension.
"""

import gzip
# import sys
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer


class Session(object):
    not_match_cnt = 0

    def __init__(self, meta_line):
        """initialize a session from metadata record"""
        assert meta_line[1] == 'M'
        self.session_id = meta_line[0]
        self.u_id = meta_line[3]
        self.day = meta_line[2]
        self.query, self.terms = None, None
        self.urls, self.clicked, self.time_passed, self.serp_id = [[] for i in range(4)]
        # self.meta_line = meta_line
        # self.query_line, self.click_line = [], []

    def add_record(self, query_line):
        """
        add query record.
        make the ids unique by adding identifiers: "q", "t", "u" and "d".
        """
        assert query_line[2] == 'Q' and query_line[0] == self.session_id
        self.query = "q" + query_line[4]
        self.terms = query_line[5].split(',')  # list of query term
        self.terms = ["t" + t for t in self.terms]

        urls = query_line[6:]
        urls = [u.split(',', 1) for u in urls]
        urls = [("u" + u, "d" + d) for u, d in urls]
        self.urls.extend(urls)  # list of [url, domain]

        serp_id = query_line[3]
        self.serp_id.extend([serp_id] * len(urls))
        self.clicked.extend([0] * len(urls))
        self.time_passed.extend([-1] * len(urls))
        # self.query_line.append(query_line)

    def add_click(self, click_line):
        assert len(self.urls) > 0 and click_line[0] == self.session_id and click_line[2] == 'C'
        url_id = "u" + click_line[4]
        serp_id = click_line[3]
        click_idx = -1

        for (i, (url, domain)) in enumerate(self.urls):
            if url == url_id and self.serp_id[i] == serp_id:
                click_idx = i
                break

        if click_idx < 0:
            type(self).not_match_cnt += 1
            # print("url in this session: \n%s\ncannot match %s" % (self.to_string(), url_id))

        else:
            self.clicked[click_idx] = 1
            self.time_passed[click_idx] = int(click_line[1])

        # self.click_line.append(click_line)

    def gen_feature(self, categories):
        """
        generate one-hot encoding feature.
        **only use log before last click position.**
        :param: `categories` that map id to index, generate from all sessions.
        :return: feature vectors X and corresponding labels Y. both in numpy.ndarray format
        """
        if sum(self.clicked) == 0:
            return None, None  # ignore sessions with no click && sessions not completed

        r_clicked = list(reversed(self.clicked))
        last_click_idx = len(r_clicked) - 1 - next(i for i, v in enumerate(r_clicked) if v > 0)  # find the index of the last click

        x = np.zeros((last_click_idx+1, len(categories)), dtype=np.int8)  # initialize a feature matrix
        y = self.clicked[: last_click_idx+1]
        y = np.asarray(y, dtype=np.int8)
        # generate feature from query term
        for t in self.terms:
            if t in categories:
                x[:, categories[t]] = 1
        # generate feature from domain
        for (indx,(u, d)) in enumerate(self.urls):
            if indx > last_click_idx:
                break
            if d in categories:
                # print("#column: ", categories[d])
                x[indx, categories[d]] = 1

        return x, y

    @staticmethod
    def gen_category(session_list, sup_thresh):
        """
        generate all categories for url, domain, query, terms.
        Currently only use `domain`, `terms` categories.
        **TODO: check if multiple call of this method return same order of mapping.
        :param:
        session_list: a list of session object.
        sup_thresh: (support threshold)integer. keep ids exist more than sup_thresh times only.
        :return: a dictionary mapping ids to index.
        """
        all_terms, all_domain = dict(), dict()
        categories = dict()
        incomplete_sess_cnt = 0
        for s in session_list:
            if not(s.terms and s.urls):
                incomplete_sess_cnt += 1
                continue
            for t in s.terms:
                all_terms[t] = all_terms.get(t, 0) + 1
            for u, d in s.urls:
                all_domain[d] = all_domain.get(d, 0) + 1
        print("#incomplete session: %d" % incomplete_sess_cnt)

        # map all ids to index
        idx = 0
        for t, v in all_terms.items():
            if v >= sup_thresh:
                categories[t] = idx
                idx += 1

        for d, v in all_domain.items():
            assert categories.get(d) is None
            if v >= sup_thresh:
                categories[d] = idx
                idx += 1

        return categories

    def gen_hash_feature(self, n_features=2 ** 10):
        """
        generate feature using hashing trick.
        :return:
        x: numpy.2darray
        y: numpy.1darray
        """
        if sum(self.clicked) == 0:
            return None, None  # ignore sessions with no click && sessions not completed

        r_clicked = list(reversed(self.clicked))
        last_click_idx = len(r_clicked) - 1 - next(i for i, v in enumerate(r_clicked) if v > 0)  # find the index of the last click
        y = self.clicked[: last_click_idx+1]
        y = np.asarray(y, dtype=np.int8)

        raw_string = []
        q_terms = self.query + ' ' + ' '.join(self.terms)
        for i in range(last_click_idx+1):
            u_d = ' '.join(self.urls[i])
            raw_string.append(q_terms + ' ' + u_d)

        # print("raw_string: %s" % raw_string)
        hv = HashingVectorizer(n_features=n_features)
        x = hv.transform(raw_string).toarray()
        return x, y

    def check_click(self, k):
        """
        check if click happens in a scaled down version of this session.
        i.e. considering top k items only.
        :param k: int between 1 and 10. The threshold position for a scaled down version of session.
        """
        assert 0 < k <= 10 and isinstance(k, int)
        if sum(self.clicked[:k]) > 0:
            return True
        return False

    def to_string(self):
        dummy = "#"*10 + "\n"
        s1 = "session_id: %s, u_id: %s, date: %s\n" % (self.session_id, self.u_id, self.day)
        s2 = "query: %s, terms: %s\n" % (self.query, self.terms)
        s3 = ""
        for (i, (url, domain)) in enumerate(self.urls):
            s3 += "pos: %-3d, url: %-10s, domain: %-10s, clicked: %d, time_passed: %-4d, serp_id: %s\n" % \
                  (i, url, domain, self.clicked[i], self.time_passed[i], self.serp_id[i])
        s = dummy+s1+s2+s3+dummy
        return s

    def show_origin_format(self):
        s = "\nmeta: " + str(self.meta_line) + "\nquery: " + str(self.query_line) + \
            "\nclick: " + str(self.click_line) + "\n"
        return s

if __name__ == '__main__':
    # print(sys.version_info)
    sessions = []
    with gzip.open('input/train.gz', 'r') as f_train:
        for (idx, line) in enumerate(f_train):
            line = line.decode('utf-8')  # decode byte to string
            line = line.strip().split('\t')

            if line[1] == 'M':
                sessions.append(Session(line))
            elif line[2] == 'Q':
                sessions[-1].add_record(line)
            elif line[2] == 'C':
                sessions[-1].add_click(line)
            else:
                raise ValueError("cannot resolve this line: \n%s" % line)

            if (idx+1) % 10000 == 0:
                break
    print("session format: \n", sessions[0].to_string())
    if Session.not_match_cnt > 0:
        print("Number of not matched url: %d" % Session.not_match_cnt)

    # feature construction (one hot)
    categories = Session.gen_category(sessions, sup_thresh=2)
    x, y = sessions[0].gen_feature(categories)
    if x is not None:
        print(x, x.shape)
        print(y, y.shape)
        print("num of non-zero elements in row of x: ", np.sum(x, axis=1))
        print("num of non-zero elements in x: ", np.sum(x))
        print()

    # another feature construction (hashing trick)
    x, y = sessions[0].gen_hash_feature(n_features=10)
    if x is not None:
        print(x, x.shape)
        print(y, y.shape)
        print()
