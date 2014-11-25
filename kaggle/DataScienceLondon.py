__author__ = 'erwin'


__author__ = 'erwin'

import os
import csv
import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


class DataScienceLondon:
    def __init__(self, base_dir='/Users/erwin/Documents/DataScienceLondon', train_set='train.csv', test_set='test2.csv'):
        self.base_dir = base_dir
        self.train_set = os.path.join(self.base_dir, train_set)
        self.train_label = os.path.join(self.base_dir, 'trainLabels.csv')
        self.test_set = os.path.join(self.base_dir, test_set)
        #self.clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
        self.clf = GradientBoostingClassifier(max_depth=3)

    def get_data(self, x_name, y_name=None):
        fin = open(x_name, 'r')
        reader = csv.reader(fin)
        x = []
        for fields in reader:
            x.append(fields)
        fin.close()

        y = []
        if not y_name is None:
            fin = open(y_name, 'r')
            for l in fin.readlines():
                l = l.rstrip()
                y.append(l)
            fin.close()

        return x, y

    def train(self, print_debug=True):
        train_x, train_y = self.get_data(self.train_set, self.train_label)

        self.clf.fit(train_x, train_y)
        if print_debug:
            test_x, test_y = self.get_data(self.train_set+"1", self.train_label+"1")
            score = self.clf.score(test_x, test_y)
            print("score: %f" % score)

    def test(self):
        test_x, _ = self.get_data(x_name=self.test_set)
        fout = open(os.path.join(self.base_dir, 'test_label.csv'), 'w')
        fout.write("Id,Solution\n")
        lineno = 1
        for t in test_x:
            y = self.clf.predict(numpy.array(t))
            fout.write("%d,%s\n" % (lineno, y[0]))
            lineno += 1
        fout.close()


if __name__ == '__main__':
    london = DataScienceLondon(test_set='test.csv')
    london.train(print_debug=True)
    london.test()