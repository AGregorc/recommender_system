import numpy as np
import csv
from scipy.sparse import bsr_matrix
from sklearn.model_selection import train_test_split


class Matrix_factorization(object):
    def __init__(self, alpha=0.01, eta=0.01, K=96, users=1892, artists=17632):
        self.alpha = alpha
        self.eta = eta
        self.K = K
        self.users = users
        self.artists = artists
        self.init_matrices()

    def init_matrices(self):
        self.P = np.random.rand(self.users, self.K) * 0.02 - 0.01
        self.P[:, 0] = np.ones(self.users)  # incorporate bias features
        self.Q = np.random.rand(self.artists, self.K) * 0.02 - 0.01
        self.Q[:, 1] = np.ones(self.artists)  # incorporate bias features

    def __call__(self, training_data, iter=1000, testing=5):
        n = len(training_data)
        rmse_avg = 0

        for t in range(testing):
            num_of_straight_worse_rmse = -1
            rmse = 0
            self.init_matrices()
            X_train, X_test = train_test_split(training_data, test_size=0.3, random_state=42)

            for i in range(iter):
                for line in X_train:
                    u = int(line[0]) - 1
                    i = int(line[1]) - 1
                    eui = (line[2] - self.P[u].dot(self.Q[i]))
                    pu = self.P[u]
                    self.P[u] = self.P[u] + self.alpha * (eui * self.Q[i] - self.eta * self.P[u])
                    self.P[u][0] = 1  # incorporate bias features
                    self.Q[i] = self.Q[i] + self.alpha * (eui * pu - self.eta * self.Q[i])
                    self.Q[i][1] = 1  # incorporate bias features

                rmse_prev = rmse
                rmse = np.sqrt(np.average(
                    [(line[2] - self.P[int(line[0]) - 1].dot(self.Q[int(line[1]) - 1])) ** 2 for line in X_test]))

                print("tmp rmse: ", rmse)
                if rmse_prev < rmse:
                    num_of_straight_worse_rmse += 1
                else:
                    num_of_straight_worse_rmse = 0
                if num_of_straight_worse_rmse > 2:
                    break
            rmse_avg += rmse
        return rmse_avg / testing

    def add_new_user(self, user_data):
        new_P_vector = np.random.rand(self.K) * 0.02 - 0.01
        for line in user_data:
            u = int(line[0]) - 1
            i = int(line[1]) - 1
            eui = (line[2] - new_P_vector.dot(self.Q[i]))
            new_P_vector = new_P_vector + self.alpha * (eui * self.Q[i] - self.eta * new_P_vector)
            new_P_vector[0] = 1  # incorporate bias features
        return new_P_vector

    def predict(self, testing_data):
        p =  np.array([self.P[int(line[0]) - 1].dot(self.Q[int(line[1]) - 1]) for line in testing_data])
        p[p < 0] = 0.1
        return p


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
error = recommender(train_data)
print("rmse average: ", error)

print('test_num: ', test_num)
write_to_file('predicted.txt', recommender.predict(test_data))
