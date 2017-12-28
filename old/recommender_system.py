import numpy as np
import csv
from scipy.sparse import bsr_matrix
from sklearn.model_selection import train_test_split


class Matrix_factorization(object):
    def __init__(self, alpha=0.01, eta=0.001, K=120, users=1892, artists=17632):
        self.alpha = alpha
        self.eta = eta
        self.K = K
        self.users = users
        self.artists = artists

        self.P = np.random.rand(self.users, self.K) * 0.02 - 0.01
        self.Q = np.random.rand(self.artists, self.K) * 0.02 - 0.01

    def __call__(self, training_data, iter=1000, testing=5):
        n = len(training_data)
        # print("n: ", n)
        num_of_straight_worse_rmse = -1
        rmse = 0
        for t in range(testing):
            X_train, X_test = train_test_split(training_data, test_size=0.1, random_state=42)

            for i in range(iter):
                for line in X_train:
                    u = int(line[0]) - 1
                    i = int(line[1]) - 1
                    eui = (line[2] - self.P[u].dot(self.Q[i]))
                    # if eui > 10:
                    #    print("big eui: ", eui, line[2], P[u].dot(Q[i]))

                    """if np.isnan(eui):
                        print("u: ", u)
                        print("P[u]: ", P[u])
                        print("i: ", i)
                        print("Q[i]: ", Q[i])
                        print(P[u].dot(Q[i]))
                        print("eui: ", eui)
                        break"""

                    # eui_reg = (eui + self.eta * P[int(line[0])-1].dot(P[int(line[0])-1]) + self.eta * Q[int(line[1])-1].dot(Q[int(line[1])-1]))
                    # print("eui: ", eui)
                    # print("eui_reg: ", eui_reg)
                    # print("fist P[u]: ", P[u])
                    pu = self.P[u]
                    self.P[u] = self.P[u] + self.alpha * (eui * self.Q[i] - self.eta * self.P[u])
                    # print("updated P[u]: ", P[u])
                    self.Q[i] = self.Q[i] + self.alpha * (eui * pu - self.eta * self.Q[i])

                    # print("calc: ", (eui * pu - self.eta * Q[i]))
                rmse_prev = rmse
                rmse = np.sqrt(np.average(
                    [(line[2] - self.P[int(line[0]) - 1].dot(self.Q[int(line[1]) - 1])) ** 2 for line in X_test]))

                print("tmp rmse: ", rmse)
                if rmse_prev < rmse:
                    num_of_straight_worse_rmse += 1
                else:
                    num_of_straight_worse_rmse = 0

                #print("num_of: ", num_of_straight_worse_rmse)
                if num_of_straight_worse_rmse > 2:
                    break

    def predict(self, testing_data):
        return np.array([self.P[int(line[0]) - 1].dot(self.Q[int(line[1]) - 1]) for line in testing_data])

def write_to_file(filename, text):
    f = open(filename, "w+")
    for line in text:
        line = str(line) + "\n"
        f.write(line)

with open('user_artists_training.dat', 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        headers = next(reader)
        train_data = np.array([(int(row[0]), int(row[1]), float(row[2])) for row in reader])
        num = train_data.max(0)
        print("max: ", num)
        print("min: ", train_data.min(0))

with open('user_artists_test.dat', 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        headers = next(reader)
        test_data = np.array([(int(row[0]), int(row[1])) for row in reader])
        test_num = test_data.max(0)
        print("max: ", test_num)
        print("min: ", test_data.min(0))


recommender = Matrix_factorization(users=int(num[0]), artists=int(num[1]) + 1)
recommender(train_data)

print('test_num: ', test_num)
write_to_file('predicted.txt',  recommender.predict(test_data))










