__author__ = 'erwin'
#coding=utf-8

'''
@ref https://www.kaggle.com/c/bike-sharing-demand/data?sampleSubmission.csv
特征工程，
'''

import os
import csv
import random
import scipy as sp
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


def RMSLE(act, pred):
    loss = sum((sp.log(sp.add(pred, 1)) - sp.log(sp.add(act, 1))) ** 2)
    loss = sp.sqrt(loss * 1.0 / len(pred))
    return loss


class BikeRental:
    def __init__(self, base_dir, train_set, test_set):
        self.base_dir = base_dir
        self.train_set = os.path.join(self.base_dir, train_set)
        self.test_set = os.path.join(self.base_dir, test_set)
        self.clf = GradientBoostingRegressor(max_depth=10)
        self.clf_rf = RandomForestRegressor(max_depth=3)

    @staticmethod
    def load_data(file_name, is_train=True, self_test=False):
        fin = open(file_name, 'r')
        reader = csv.reader(fin)
        train_X = []
        train_y = []
        test_X = []
        test_y = []
        reader.next()                       # skip title
        lineno = 0
        for fields in reader:
            lineno += 1
            origin = fields[0]
            month = int(fields[0][5:7])
            day = int(fields[0][8:10])
            fields[0] = fields[0][11:13]    # only get hour in this day as a feature
            if self_test and lineno % random.randint(20, 50) == 0:
                X = test_X
                y = test_y
            else:
                X = train_X
                y = train_y

            for i in range(len(fields)):
                if fields[i] == '':
                    fields[i] = 0
                    continue
                if fields[i].find('.') > -1:
                    fields[i] = float(fields[i])
                else:
                    fields[i] = int(fields[i])

            feature = [month]
            if is_train:
                X.append(feature + fields[:-3])
                y.append(fields[-1])
            else:
                X.append(feature + fields)
                y.append(origin)
        fin.close()
        return train_X, train_y, test_X, test_y

    def train(self, self_test=True):
        train_X, train_y, test_X, test_y = self.load_data(self.train_set, is_train=True, self_test=self_test)
        self.clf.fit(train_X, train_y)
        self.clf_rf.fit(train_X, train_y)

        if self_test:
            score = self.clf.score(test_X, test_y)
            score_rf = self.clf_rf.score(test_X, test_y)

            pred_y = []
            pred_rf_y = []
            fout = open(os.path.join(self.base_dir, 'train_test.csv'), 'w')
            fout.write("datetime,season,holiday,workingday,weather,temp,atemp,humidity,windspeed,count,predict, pred_rf\n")
            for t, act_y in zip(test_X, test_y):
                f = ','.join(map(lambda i: str(i), t))
                y = self.clf.predict(t)
                y_rf = self.clf_rf.predict(t)
                if y[0] >= 0:
                    pred_y.append(y[0])
                else:
                    pred_y.append(0)
                if y_rf >= 0:
                    pred_rf_y.append(y_rf[0])
                else:
                    pred_rf_y.append(0)
                fout.write("%s,%d,%d,%d\n" % (f, act_y, y[0], y_rf[0]))
            fout.close()
            loss = RMSLE(test_y, pred_y)
            loss_rf = RMSLE(test_y, pred_rf_y)
            print("score: %f, %f; RMSLE: %f, %f" % (score, score_rf, loss, loss_rf))

    def test(self):
        fout = open(os.path.join(self.base_dir, 'test_out.csv'), 'w')
        fout.write("datetime,count\n")
        test_X, origin, _, _ = self.load_data(self.test_set, is_train=False)
        for t, orig in zip(test_X, origin):
            y = self.clf.predict(X=t)
            if y < 0:
                y = 0
            fout.write("%s,%d\n" % (orig, y))
        fout.close()

if __name__ == '__main__':
    bike_rental = BikeRental('/Users/erwin/Documents/BikeRental', 'train.csv', 'test.csv')
    bike_rental.train(self_test=True)
    #bike_rental.test()