"""preprocess yandex data set for personalized web search"""
# python3.5

import gzip
import sys
import numpy as np


class Session(object):
    notMatchCnt = 0

    def __init__(self, metaLine):
        """init a session from metadata record"""
        assert metaLine[1] == 'M'
        self.session_id = metaLine[0]
        self.u_id = metaLine[3]
        self.day = metaLine[2]
        self.query, self.terms, self.urls, self.clicked = (None,)*4

    def add_record(self, queryLine):
        assert queryLine[2] == 'Q' and queryLine[0] == self.session_id
        self.query = queryLine[4]
        self.terms = queryLine[5].split(',')  # list of query term
        urls = queryLine[6:]
        urls = [u.split(',', 1) for u in urls]
        self.urls = urls  # list of [url, domain]
        self.clicked = [0] * len(self.urls)

    def add_click(self, clickLine):
        assert len(self.urls) > 0 and clickLine[0] == self.session_id and clickLine[2] == 'C'
        url_id = clickLine[-1]
        idx = -1
        for (i, (url, domain)) in enumerate(self.urls):
            if url == url_id:
                idx = i
                break
        if idx < 0:
            type(self).notMatchCnt += 1
            # print("url in this session: \n%s\ncannot match %s" % (self.to_string(), url_id))
            # raise ValueError("cannot find matched url_id: %s" % url_id)
        self.clicked[idx] = 1

    def gen_feature(self):
        """
        generate one-hot encoding feature.
        **only use log before last click position.**
        :return: feature vectors X and corresponding labels Y. both in numpy.ndarray format
        """

        pass

    @staticmethod
    def gen_category():
        """
        generate all categories for url, domain, query, terms.
        Currently only use `domain`, `terms` categories.
        """
        pass

    def nDCG(self):
        """evaluation metric"""
        pass

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
        s1 = "session_id: %s\tu_id: %s\n" % (self.session_id, self.u_id)
        s2 = "query: %s, terms: %s\n" % (self.query, self.terms)
        s3 = ""
        for (i,(url, domain)) in enumerate(self.urls):
            s3 += "pos: %d, url: %s, domain: %s, clicked: %d\n" % (i,url,domain, self.clicked[i])
        s = dummy+s1+s2+s3+dummy
        return s

if __name__ == '__main__':
    # print(sys.version_info)
    sessions = []
    with gzip.open('input/train.gz', 'r') as f_train:
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

            if (idx+1) % 1000 == 0:
                print(sessions[-1].to_string())
                break
        print("Number of not matched url: %d" % Session.notMatchCnt)
