__author__ = 'erwin'

import os
import csv
import numpy
from sklearn.ensemble import RandomForestClassifier

class PredictTitanic:
    def __init__(self, base_dir='/Users/erwin/Documents/Titanic Disatar', train_set='train.csv', test_set='test2.csv'):
        self.base_dir = base_dir
        self.train_set = os.path.join(base_dir, train_set)
        self.test_set = os.path.join(base_dir, test_set)
        self.clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)

    def     get_data(self, file_name, start=2):
        fin = open(file_name, 'r')
        reader = csv.reader(fin)
        fields = reader.next()
        x = []
        y = []
        for fields in reader:
            try:
                pclass = int(fields[start])
                if fields[start+2] == 'male':
                    sex = 1
                else:
                    sex = 0
                age = float(fields[start+3]) if len(fields[start+3]) > 0 else 0
                sibsp = int(fields[start+4])
                parch = int(fields[start+5])
                fare = float(fields[start+7]) if len(fields[start+7]) > 0 else 0
                embarked = 0    # 'C'
                if fields[start+9] == 'Q':
                    embarked = 1
                elif fields[start+9] == 'S':
                    embarked = 2
                tmp = [pclass, sex, age, sibsp, parch, fare, embarked]
                print(tmp)
                x.append(tmp)
                if start == 2:
                    y.append(int(fields[1]))
            except KeyError:
                print("error at: ", fields)
            except ValueError:
                print("error at: ", fields)
        fin.close()
        return x, y

    def train(self, print_debug=True):
        train_x, train_y = self.get_data(self.train_set)
        self.clf.fit(train_x, train_y)
        if print_debug:
            test_x, test_y = self.get_data(self.test_set)

            score = self.clf.score(test_x, test_y)

            print("score: %f" % score)

    def test(self):
        test_x, test_y = self.get_data(self.test_set, start=1)
        fout = open(os.path.join(self.base_dir, 'test_out.csv'), 'w')
        fout.write("PassengerId,Survived\n")
        id = 892
        for t in test_x:
            y = self.clf.predict(numpy.array(t))
            fout.write("%d,%d\n" % (id, y))
            id += 1
        fout.close()


if __name__ == '__main__':
    titanic = PredictTitanic(test_set='test.csv')
    titanic.train(print_debug=False)
    titanic.test()
