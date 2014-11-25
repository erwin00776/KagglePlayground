__author__ = 'erwin'

#coding=utf-8

import pickle
import os
import numpy as np
import random
from scipy import *
from scipy import linalg
np.seterr(divide='ignore', invalid='ignore')


class MovieRatings:
    '''
    By using conventional SVD, decompose the sparse user-movie matrix directly.
    '''
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.users = {}
        self.users_avg = {}
        self.movies = {}
        self.matrix = None
        self.U, self.Sigma, self.VT = None, None, None
        self.user_U, self.user_Sigma, self.user_VT = None, None, None
        self.tmpfile = '/tmp/movie.svd.tmp'
        self.test_set = []
        self.test_set_orig = []
        self.index2user = []
        self.index2movie = []
        self.__load__()

    @staticmethod
    def cosine_similarity(va, vb):
        a = float(va.T * vb)
        b = linalg.norm(va) * linalg.norm(vb)
        if b == 0:
            return 0.5
        return 0.5 + 0.5 * a / b

    def build(self):
        train_fin = open(self.train_path, 'r')
        train_fin.readline()    # skip title
        user_index = {}
        movie_index = {}
        lineno = 0
        for line in train_fin.readlines():
            lineno += 1
            line = line.rstrip()
            fields = line.split(',')
            user = fields[1]
            movie = fields[2]
            rating = int(fields[3])
            if user in self.users:
                self.users[user].append([movie, rating])
            else:
                user_index[user] = len(self.users)
                self.index2user.append(user)
                self.users[user] = [(movie, rating)]
            if movie in self.movies:
                self.movies[movie].append([user, rating])
            else:
                movie_index[movie] = len(self.movies)
                self.index2movie.append(movie)
                self.movies[movie] = [(user, rating)]
        train_fin.close()

        test_fin = open(self.test_path, 'r')
        lineno = 0
        for line in test_fin.readlines():
            line = line.rstrip()
            lineno += 1
            fields = line.split(',')
            user_idx = user_index.get(fields[1], None)
            movie_idx = movie_index.get(fields[2], None)
            if user_idx is None or movie_idx is None:
                continue
            rating = int(fields[3])
            self.test_set.append([user_idx, movie_idx, rating])
            self.test_set_orig.append([fields[1], fields[2], rating])
        test_fin.close()

        self.matrix = np.zeros([len(self.users), len(self.movies)])
        for (user, movie_list) in self.users.items():
            uidx = user_index[user]
            for (movie, rating) in movie_list:
                midx = movie_index[movie]
                self.matrix[uidx, midx] = rating

        count = 0
        for (user, ratings) in self.users.items():
            s = sum(map(lambda (x, y): y, ratings))
            self.users_avg[user] = s * 1.0 / len(ratings)
            count += 1

    def __load__(self):
        if not self.U is None or not os.path.exists(self.tmpfile):
            return
        fin = open(self.tmpfile, 'rb')
        self.U = pickle.load(fin)
        self.Sigma = pickle.load(fin)
        self.VT = pickle.load(fin)
        fin.close()

        if not self.user_U is None or not os.path.exists(self.tmpfile+"_user"):
            return
        fin_user = open(self.tmpfile + "_user", 'rb')
        self.user_U = pickle.load(fin_user)
        self.user_Sigma = pickle.load(fin_user)
        self.user_VT = pickle.load(fin_user)
        fin_user.close()

    def __save__(self):
        fout = open(self.tmpfile, 'wb')
        pickle.dump(self.U, fout)
        pickle.dump(self.Sigma, fout)
        pickle.dump(self.VT, fout)
        fout.close()

        fout_user = open(self.tmpfile + "_user", 'wb')
        pickle.dump(self.user_U, fout_user)
        pickle.dump(self.user_Sigma, fout_user)
        pickle.dump(self.user_VT, fout_user)
        fout_user.close()

    def svd_est_user(self, user_idx=None, movie_idx=None):
        if self.user_U is None:
            self.user_U, self.user_Sigma, self.user_VT = linalg.svd(self.matrix.T)
            self.__save__()
        features = 1
        Sig4 = mat(eye(features) * self.Sigma[:features])
        tmp = self.user_U[:, :features] * Sig4.I
        # find the similarity users
        users = self.matrix * tmp

        # count of all users
        n = shape(self.matrix)[0]
        sim_sum = 0
        sim_total = 0
        #user_avg = self.users_avg[self.index2user[user_idx]]
        for i in range(n):
            user_rating = self.matrix[i, movie_idx]
            if user_rating == 0 or i == user_idx:
                continue
            #sim_user_avg = self.users_avg[self.index2user[i]]
            sim = self.cosine_similarity(users[user_idx, :].T, users[i, :].T)
            sim_total += 1
            #sim_sum += user_rating * (user_avg / sim_user_avg) * sim
            sim_sum += user_rating * sim
        if sim_total == 0:
            return 0
        else:
            return sim_sum / sim_total

    def svd_est(self, user_idx=None, movie_idx=None, sim_method=cosine_similarity):
        '''
        item based
        :param user_idx:
        :param movie_idx:
        :param sim_method:
        :return:
        '''
        if self.U is None:
            self.U, self.Sigma, self.VT = linalg.svd(self.matrix)
            self.__save__()
        features = 1
        Sig4 = mat(eye(features) * self.Sigma[:features])
        tmp = self.U[:, :features] * Sig4.I
        movies = self.matrix.T * tmp

        # count of movies
        n = shape(self.matrix)[1]
        sim_sum = 0
        sim_total = 0
        for i in range(n):
            user_rating = self.matrix[user_idx, i]
            if user_rating == 0 or i == movie_idx:
                continue
            sim = self.cosine_similarity(movies[movie_idx, :].T, movies[i, :].T)
            sim_total += 1
            sim_sum += user_rating * sim
        if sim_total == 0:
            return 0
        else:
            return sim_sum / sim_total

    def test(self):
        diffs = 0.0
        diffs_user = 0.0
        diffs_avg = 0.0
        diffs_round = 0.0
        i = 0
        for t in self.test_set:
            if t[0] is None or t[1] is None:
                print("bad test data!!!")
                continue
            sim = self.svd_est(user_idx=t[0], movie_idx=t[1])
            sim_user = self.svd_est_user(user_idx=t[0], movie_idx=t[1])
            sim_avg = (sim + sim_user) / 2
            diffs += (sim - t[2]) ** 2
            diffs_user += (sim_user - t[2]) ** 2
            diffs_avg += (sim_avg - t[2]) ** 2
            diffs_round += (round(sim_avg) - t[2]) ** 2
            orig = self.test_set_orig[i]
            print("test result: %s\t%s\t\t%d\t\t%f\t%f\t%f" %
                  (orig[0] , orig[1], orig[2],
                   sim, sim_user, sim_avg))
            i += 1
            if i > 2000:
                break
        print(diffs, i)
        print(sqrt(diffs/i), sqrt(diffs_user/i), sqrt(diffs_avg/i), sqrt(diffs_round/i))


class AvgRating:
    def __init__(self):
        self.sum = 0
        self.count = 0

    def add(self, s):
        self.sum += s
        self.count += 1


class ContinuesMovieRatings:
    def __init__(self, train_set, test_set):
        self.base_dir = '/Users/erwin/Documents/Predict Movie Ratings/'
        self.train_set = os.path.join(self.base_dir, train_set)
        self.test_set = os.path.join(self.base_dir, test_set)

        self.bias_u, self.bias_i, self.pu, self.qi = None, None, None, None
        self.user_count, self.item_count = 0, 0
        self.score_avg = 3.0
        self.factor_num, self.learn_rate, self.regularization = 5, 0.005, 0.01

        self.users2id, self.movies2id = None, None
        self.users_avg, self.movies_avg = {}, {}

        self.debug_info = {}

        self.__load__()
        self.prepare_data()

    def __dump__(self):
        path = os.path.join(self.base_dir, 'continues.model')
        fout = open(path, 'wb')
        pickle.dump(self.bias_u, fout)
        pickle.dump(self.bias_i, fout)
        pickle.dump(self.pu, fout)
        pickle.dump(self.qi, fout)
        fout.close()

    def __load__(self):
        path = os.path.join(self.base_dir, 'continues.model')
        if not os.path.exists(path):
            return
        fin = open(path, 'rb')
        self.bias_u = pickle.load(fin)
        self.bias_i = pickle.load(fin)
        self.pu = pickle.load(fin)
        self.qi = pickle.load(fin)
        fin.close()

    @staticmethod
    def __prepare__(filename, users2id={}, movies2id={}, train=True):
        fin = open(filename, 'r')
        fout = open(filename + '.prepared', 'w')
        id2users = []
        id2movies = []
        users_avg = {}
        movies_avg = {}
        for line in fin.readlines():
            line = line.rstrip()
            fields = line.split(',')
            user_id = users2id.get(fields[1], None)
            if user_id is None:
                user_id = len(users2id)
                users2id[fields[1]] = user_id
                id2users.append(fields[1])
            movie_id = movies2id.get(fields[2], None)
            if movie_id is None:
                movie_id = len(movies2id)
                movies2id[fields[2]] = movie_id
                id2movies.append(fields[2])
            rating = 0
            if train:
                rating = int(fields[3])

                uavg = users_avg.get(fields[1], None)
                if uavg is None:
                    uavg = AvgRating()
                    users_avg[fields[1]] = uavg
                uavg.add(rating)
                iavg = movies_avg.get(fields[2], None)
                if iavg is None:
                    iavg = AvgRating()
                    movies_avg[fields[2]] = iavg
                iavg.add(rating)

            fout.write("%d,%d,%d\n" % (user_id, movie_id, rating))
        fout.close()
        fin.close()
        if train:
            return users2id, movies2id, id2users, id2movies, users_avg, movies_avg
        else:
            return users2id, movies2id, id2users, id2movies

    def prepare_data(self):
        users2id = None
        movies2id = None
        id2users = None
        id2movies = None
        if not os.path.exists(self.train_set + '.prepared'):
            users2id, movies2id, id2users, id2movies, self.users_avg, self.movies_avg = self.__prepare__(self.train_set)
            #self.__prepare__(self.test_set, users2id, movies2id, train=False)
            fout = open(os.path.join(self.base_dir, 'map.data'), 'wb')
            pickle.dump(users2id, fout)
            pickle.dump(movies2id, fout)
            pickle.dump(id2users, fout)
            pickle.dump(id2movies, fout)
            pickle.dump(self.users_avg, fout)
            pickle.dump(self.movies_avg, fout)
            fout.close()
        else:
            fin = open(os.path.join(self.base_dir, 'map.data'), 'rb')
            users2id = pickle.load(fin)
            movies2id = pickle.load(fin)
            id2users = pickle.load(fin)
            id2movies = pickle.load(fin)
            self.users_avg = pickle.load(fin)
            self.movies_avg = pickle.load(fin)
            fin.close()
        self.user_count = len(users2id)
        self.item_count = len(movies2id)
        self.users2id = users2id
        self.movies2id = movies2id

    @staticmethod
    def inner_product(pu, qi):
        s = 0
        for i in range(len(pu)):
            s += pu[i] * qi[i]
        return s

    def predict(self, av, bu, bi, pu, qi):
        score = av + bu + bi + self.inner_product(pu, qi)
        if score < 1:
            return 1.0
        elif score > 5:
            return 5.0
        return score

    def train(self):
        if not (self.bias_i is None and self.bias_u is None):
            return
        self.bias_i = [0.0 for i in range(self.item_count)]
        self.bias_u = [0.0 for i in range(self.user_count)]
        temp = sqrt(self.factor_num)
        self.pu = [[(0.1 * random.random() / temp) for j in range(self.factor_num)] for i in range(self.user_count)]
        self.qi = [[(0.1 * random.random() / temp) for j in range(self.factor_num)] for i in range(self.item_count)]

        iterations = 35
        self.debug_info['iter'] = []
        for step in range(iterations):
            fin = open(self.train_set + '.prepared', 'r')
            eui_sum = 0
            eui_count = 0
            for line in fin.readlines():
                line = line.rstrip()
                fields = line.split(',')
                uid, iid, rating = int(fields[0]), int(fields[1]), int(fields[2])
                prediction = self.predict(self.score_avg, self.bias_u[uid], self.bias_i[iid],
                                          self.pu[uid], self.qi[iid])
                eui = prediction - rating
                eui_sum += abs(eui)
                eui_count += 1
                self.bias_u[uid] -= self.learn_rate * (eui - self.regularization * self.bias_u[uid])
                self.bias_i[iid] -= self.learn_rate * (eui - self.regularization * self.bias_i[iid])
                for k in range(self.factor_num):
                    temp = self.pu[uid][k]
                    self.pu[uid][k] -= self.learn_rate * (eui * self.qi[iid][k] - self.regularization * self.pu[uid][k])
                    self.qi[iid][k] -= self.learn_rate * (eui * temp - self.regularization * self.qi[iid][k])
            print(self.bias_u)
            print('eui changed %d' % (eui_sum * 1000 / eui_count))
            self.debug_info['iter'].append(eui_sum * 1000 / eui_count)
            if eui_sum * 1000 / eui_count < 600:
                break
        self.__dump__()

    def validate(self):
        if not os.path.exists(self.test_set + '.prepared'):
            self.__prepare__(self.test_set, self.users2id, self.movies2id, train=True)
        fin = open(self.test_set + '.prepared', 'r')
        count = 0
        diffs = 0.0
        for line in fin.readlines():
            line = line.rstrip()
            fields = line.split(',')
            uid, iid, rating = int(fields[0]), int(fields[1]), int(fields[2])
            prediction = self.predict(self.score_avg, self.bias_u[uid], self.bias_i[iid],
                                      self.pu[uid], self.qi[iid])
            print("test: %d \t %d \t %d \t %f" % (uid, iid, rating, prediction))
            diffs += (prediction - rating) ** 2
            count += 1
        print("%f" % sqrt(diffs/count))
        fin.close()

    def test(self):
        fin = open(self.test_set, 'r')
        fout = open(os.path.join(self.base_dir, 'output.csv'), 'w')
        fout.write("ID,rating\n")
        # skip line
        fin.readline()
        nonexists = 0
        for line in fin.readlines():
            line = line.rstrip()
            fields = line.split(',')
            uid = self.users2id.get(fields[1], None)
            iid = self.movies2id.get(fields[2], None)
            if not(uid is None and iid is None):
                #print(self.user_count, self.item_count, uid, iid)
                prediction = self.predict(self.score_avg, self.bias_u[uid], self.bias_i[iid],
                                          self.pu[uid], self.qi[iid])
            elif uid is None and iid is None:
                prediction = 3.0
                nonexists += 1
            elif uid is None and not iid is None:
                avg = self.movies_avg.get(fields[2])
                prediction = round(avg.sum / avg.count)
                print('uid is None')
            else:
                avg = self.users_avg.get(fields[1])
                prediction = round(avg.sum / avg.count)
                print('iid is None')
            fout.write("%s,%d\n" % (fields[0], int(prediction)))
        fout.close()
        fin.close()

        print("non exists count %d" % nonexists)

    def print_debug(self):
        i = 0
        for eui in self.debug_info['iter']:
            print('%02d eui changed %d' % (i, eui))
            i += 1


if __name__ == "__main__":
    '''
    movie_ratings = MovieRatings(train_path='/Users/erwin/Documents/Predict Movie Ratings/train_v2_5w.csv',
                                 test_path='/Users/erwin/Documents/Predict Movie Ratings/test_v2_5k.csv')
    movie_ratings.build()
    movie_ratings.test()
    '''
    cmr = ContinuesMovieRatings('train_v2.csv', 'test_v2.csv')
    cmr.train()
    #cmr.validate()
    #cmr.print_debug()
    cmr.test()