__author__ = 'erwin'
#coding=utf-8

import os
import json
import random
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as sk_m
#from sklearn.linear_model import
from sklearn import svm
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation

class RandomActsOfPizza:
    def __init__(self,
                 base_dir='/Users/erwin/Documents/RandomActsOfPizza',
                 train_set='train.json',
                 test_set='test.json'):
        self.base_dir = base_dir
        self.train_set = os.path.join(self.base_dir, train_set)
        self.test_set = os.path.join(self.base_dir, test_set)
        self.clf = RandomForestClassifier(max_depth=5, n_estimators=10)
        self.clf_svm = svm.SVC(kernel='rbf')

    @staticmethod
    def __parse__(file_name, is_train=False, self_test=False):
        tmp_name = os.path.join('/tmp', os.path.basename(file_name))
        if os.path.exists(tmp_name):
            fin = open(tmp_name, 'rb')
            train_X = pickle.load(fin)
            train_y = pickle.load(fin)
            test_X = pickle.load(fin)
            test_y = pickle.load(fin)
            fin.close()
            return train_X, train_y, test_X, test_y

        fin = open(file_name, 'r')
        dat = fin.readlines()
        fin.close()
        dat = ''.join(dat)
        case_list = json.loads(dat)
        X, y = None, None
        train_X, train_y, test_X, test_y = [], [], [], []
        lineno = 0
        
        subs = {}
        for case in case_list:
            for sub in case['requester_subreddits_at_request']:
                if not sub in subs:
                    subs[sub] = 1
                else:
                    subs[sub] += 1

        for case in case_list:
            if self_test and (lineno % random.randint(10, 15)) == 0:
                X, y = test_X, test_y
            else:
                X, y = train_X, train_y
            lineno += 1

            request_id = case['request_id']
            if not is_train:
                test_X.append(request_id)
            title_len = len(case['request_title'].split())
            if 'request_text_edit_aware' in case:
                text_len = len(case['request_text_edit_aware'].split())
            else:
                text_len = len(case['request_text'].split())
            giver = 0 if case['giver_username_if_known'] == 'N/A' else 1
            age = int(float(case['requester_account_age_in_days_at_request']))
            days = int(case['requester_days_since_first_post_on_raop_at_request'])
            comment1 = int(case['requester_number_of_comments_at_request'])
            comment2 = int(case['requester_number_of_comments_in_raop_at_request'])
            post1 = int(case['requester_number_of_posts_at_request'])
            post2 = int(case['requester_number_of_posts_on_raop_at_request'])
            subreddits = int(case['requester_number_of_subreddits_at_request'])
            up_m_down = int(case['requester_upvotes_minus_downvotes_at_request'])
            up_p_down = int(case['requester_upvotes_plus_downvotes_at_request'])
            timestamp = int(float(case['unix_timestamp_of_request_utc']))

            subs_set = set(case['requester_subreddits_at_request'])
            subs_features = []
            for (sub, c) in subs.items():
                if c < 2:
                    continue
                if sub in subs_set:
                    subs_features.append(1)
                else:
                    subs_features.append(0)

            #features = [title_len, text_len, giver, age, days, comment1, comment2, post1, post2, subreddits, up_m_down, up_p_down, timestamp] + subs_features
            features = [giver, age, days, comment1, comment2, post1, post2, subreddits, up_m_down, up_p_down, timestamp]
            X.append(features)
            if 'requester_received_pizza' in case:
                got = 1 if case['requester_received_pizza'] else 0
                y.append(got)

        fout = open(tmp_name, 'wb')
        pickle.dump(train_X, fout)
        pickle.dump(train_y, fout)
        pickle.dump(test_X, fout)
        pickle.dump(test_y, fout)
        fout.close()

        return train_X, train_y, test_X, test_y

    def train(self, self_test, my_t=1):
        train_X, train_y, test_X, test_y = self.__parse__(self.train_set, is_train=True, self_test=self_test)

        train_svm_y = map(lambda i: 1 if i == 1 else -1, train_y)
        test_svm_y = map(lambda i: 1 if i == 1 else -1, test_y)
        self.clf.fit(train_X, train_y)
        self.clf_svm.fit(train_X, train_svm_y)

        train_score = cross_validation.cross_val_score(self.clf_svm, train_X, train_y, cv=5)

        if self_test:
            score = self.clf.score(test_X, test_y)
            score_svm = self.clf_svm.score(test_X, test_svm_y)
            print("self test score(%d): %f, %f, %f" % (len(test_X),
                                                       score,
                                                       score_svm,
                                                       train_score.mean()))

        if my_t == 1:
            error_count1, error_count2, error_count3 = 0, 0, 0
            count2, count3 = 0, 0

            for x, y in zip(train_X, train_y):
                y1 = self.clf.predict(x)
                if y1 != y:
                    error_count1 += 1
            pred_y = []
            pred_y_svm = []
            for x, y in zip(test_X, test_y):
                y1 = self.clf.predict(x)
                y1_svm = self.clf_svm.predict(x)
                pred_y.append(y1)
                pred_y_svm.append(y1_svm)
                if y == 1:
                    count2 += 1
                else:
                    count3 += 1
                if y1 != y and y == 1:
                    error_count2 += 1
                elif y1 != y and y == 0:
                    error_count3 += 1
            print("error rate:(train) %f(%d), (test-1) %f(%d), (test-0) %f(%d)" % (
                error_count1 * 1.0 / len(train_X),
                len(train_X),
                error_count2 * 1.0 / count2,
                count2,
                error_count3 * 1.0 / count3,
                count3))
            score = sk_m.metrics.roc_auc_score(test_y, pred_y)
            pred_y_svm = map(lambda i: 0 if i == -1 else 1, pred_y_svm)
            score_svm = sk_m.metrics.roc_auc_score(test_y, pred_y_svm)
            print("auc %f, %f" % (score, score_svm))

    def test(self):
        fout = open(os.path.join(self.base_dir, 'test_out.txt'), 'w')
        fout.write("request_id,requester_received_pizza\n")
        test_X, test_y, request_ids, _ = self.__parse__(self.test_set, is_train=False, self_test=False)
        for request_id, t in zip(request_ids, test_X):
            score = self.clf.predict(t)
            fout.write('%s,%d\n' % (request_id, score))
        fout.close()


if __name__ == '__main__':
    pizza = RandomActsOfPizza()
    pizza.train(self_test=True, my_t=1)
    #pizza.test()