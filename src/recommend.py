from numpy.lib.function_base import append
import torch
import argparse
import numpy as np
from data_loader import load_data
from train import get_feed_dict
from model import RippleNet

np.random.seed(555)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='movie', help='which dataset to use')
parser.add_argument('--dim', type=int, default=16, help='dimension of entity and relation embeddings')
parser.add_argument('--n_hop', type=int, default=2, help='maximum hops')
parser.add_argument('--kge_weight', type=float, default=0.01, help='weight of the KGE term')
parser.add_argument('--l2_weight', type=float, default=1e-7, help='weight of the l2 regularization term')
parser.add_argument('--lr', type=float, default=0.02, help='learning rate')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--n_epoch', type=int, default=10, help='the number of epochs')
parser.add_argument('--n_memory', type=int, default=32, help='size of ripple set for each hop')
parser.add_argument('--item_update_mode', type=str, default='plus_transform',
                    help='how to update item at the end of each hop')
parser.add_argument('--using_all_hops', type=bool, default=True,
                    help='whether using outputs of all hops or just the last hop when making prediction')
parser.add_argument('--model_dir', type=str, default='../model', help='directory to save model')
parser.add_argument('--use_cuda', type=bool, default=False, help='whether to use gpu')
parser.add_argument('--k', type=int, default=10, help='number of top_k')
args = parser.parse_args()

show_loss = False
data_info = load_data(args)
print(data_info[3:5])
# train(args, data_info, show_loss)

# train_data = data_info[0]
# eval_data = data_info[1]
# test_data = data_info[2]
n_entity = data_info[3]
n_relation = data_info[4]
ripple_set = data_info[5]

user2hash = dict()
for line in open('../data/steam/user_hash.tsv', encoding='utf-8').readlines():
    k = line.strip().split('\t')[0]
    v = line.strip().split('\t')[1]
    user2hash[k] = int(v)

item2hash = dict()
for line in open('../data/steam/item_hash.tsv', encoding='utf-8').readlines():
    k = line.strip().split('\t')[0]
    v = line.strip().split('\t')[1]
    item2hash[k] = int(v)
hash2item = list(item2hash.keys())

import csv
game_path = '../data/steam/steam.csv'
game_csv = open(game_path, encoding='utf-8')
game_csv.readline()
reader = csv.reader(game_csv)
steam_csv = {}
for array in reader:
    # genre, developer, publisher, year
    steam_csv[array[1]] = [array[9], array[4], array[5], array[2].split('-')[0]]

import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtSql import QSqlDatabase, QSqlQuery
from PyQt5.QtCore import Qt, QSize, QFileInfo, QTimer, QDateTime

class History(QListWidget):
    def __init__(self, parent=None, link=None):
        super(History, self).__init__(parent)
        self.his = []
        self.user = parent.user
        file_list = ['../data/steam/ratings.tsv', '../data/steam/ratings_append.tsv']
        for file in file_list:
            for line in open(file, encoding='utf-8').readlines():
                u = line.strip().split('\t')[0]
                i = line.strip().split('\t')[1]
                if u == self.user:
                    self.his.append(i)
                    print('{:10d} {}'.format(item2hash[i], i))
                    child = QListWidgetItem()
                    child.setText(i)
                    self.addItem(child)

# 66755246
class Recommendation(QTableWidget):
    def __init__(self, parent=None, link=None, load=None):
        super(Recommendation, self).__init__(parent)
        # self.history = link
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setHorizontalScrollMode(QListWidget.ScrollPerPixel)
        self.setColumnCount(7)
        self.setHorizontalHeaderLabels(['标题','类别','制作商','发行商','发行时间','推荐指数','买'])
        self.user = parent.user
        user = self.user
        model = RippleNet(args, n_entity, n_relation)
        state_dict = torch.load('../ripplenet.pth')
        model.load_state_dict(state_dict)
        model.eval()

        self.scores = []
        data = np.array([[user2hash[user], 0, 0]])
        feed_data = get_feed_dict(args, model, data, ripple_set, 0, 1)
        # LEN = len(item2hash)
        # hash2item = list(item2hash.values())
        # for i in range(LEN):
        #         load.setWindowTitle('生成推荐中……({}%)'.format(int(i/LEN)))
        #         v = hash2item[i]
        for v in item2hash.values():
            # if v not in history:
                feed_data[0][0] = v
                return_dict = model(*feed_data)
                self.scores.append(return_dict['scores'].cpu().detach().item())
        # while start < data.shape[0]:
        #     return_dict = model(*)
        #     self.scores.append(return_dict['scores'].cpu().detach().numpy().tolist())
        print('prediction finished.')

        self.scores = np.array(self.scores).reshape(-1)

        self.MAXLEN = 3000
        self.scores_indices = (-self.scores).argsort()[:self.MAXLEN]
        np.random.shuffle(self.scores_indices)

        # writer = open('../recommendation_for_{}.tsv'.format(user), 'w', encoding='utf-8')
        # for i in range(len(self.scores_indices)):
        # self.setColumnCount(1)

        i = 0
        idx = 0
        while i < 100:
            # writer.write('{:10d}\t{}\t{}\n'.format(self.scores_indices[i], hash2item[self.scores_indices[i]], self.scores[self.scores_indices[i]]))

            if hash2item[self.scores_indices[idx]] not in self.history.his and self.scores[self.scores_indices[idx]] > 0.5:
                print('{:10d} {}'.format(self.scores_indices[idx], hash2item[self.scores_indices[idx]]))

                self.setRowCount(i+1)
                item = QTableWidgetItem(hash2item[self.scores_indices[idx]])
                self.setItem(i, 0, item)

                item = QTableWidgetItem(steam_csv[hash2item[self.scores_indices[idx]]][0].replace(';','\n'))
                self.setItem(i, 1, item)
                for j in range(1,4):
                    item = QTableWidgetItem(steam_csv[hash2item[self.scores_indices[idx]]][j])
                    self.setItem(i, 1+j, item)

                item = QTableWidgetItem('{:.2f}'.format(float(self.scores[self.scores_indices[idx]])))
                self.setItem(i, 5, item)

                btn = QPushButton('🛒')
                btn.clicked.connect(self.make_onClicked(i, hash2item[self.scores_indices[idx]]))
                self.setCellWidget(i, 6, btn)
                i += 1

            if idx < self.MAXLEN:
                idx += 1
            else:
                break
        self.tail = idx

    def make_onClicked(self, i, game):
        def onClicked():
            child = QListWidgetItem()
            child.setText(game)
            self.history.addItem(child)
            self.history.his.append(hash2item[self.scores_indices[self.tail]])

            while self.tail < self.MAXLEN:
                if hash2item[self.scores_indices[self.tail]] not in self.history.his:
                    item = QTableWidgetItem(hash2item[self.scores_indices[self.tail]])
                    self.setItem(i, 0, item)
                    for j in range(4):
                        item = QTableWidgetItem(steam_csv[hash2item[self.scores_indices[self.tail]]][j])
                        self.setItem(i, 1+j, item)
                    item = QTableWidgetItem('{:.2f}'.format(float(self.scores[self.scores_indices[self.tail]])))
                    self.setItem(i, 5, item)
                    btn = QPushButton('🛒')
                    btn.clicked.connect(self.make_onClicked(i, hash2item[self.scores_indices[self.tail]]))
                    self.setCellWidget(i, 6, btn)
                    break
                self.tail += 1
            appender = open('../data/steam/ratings_append.txt', 'a', encoding='utf-8')
            appender.write('{}\t{}\t1\n'.format(user2hash[self.user], item2hash[game]))
            appender.close()
            appender = open('../data/steam/ratings_append.tsv', 'a', encoding='utf-8')
            appender.write('{}\t{}\t1\n'.format(self.user, game))
            appender.close()
        return onClicked


class Window(QSplitter):
    def __init__(self, parent=None, link=None):
        super(Window, self).__init__(parent)

        self.setWindowTitle('游戏推荐系统')
        self.user = link.user
        self.history = History(self)
        self.recommendation = Recommendation(self, link=self.history, load=link)
        self.addWidget(self.history)
        self.addWidget(self.recommendation)
        self.setOrientation(Qt.Horizontal)
        self.setSizes([200, 800])
        self.resize(1000, 500)

    def closeEvent(self, event):
        pass

class Login(QWidget):
    def __init__(self, parent=None, link=None):
        super(Login, self).__init__(parent)
        layout = QGridLayout(self)
        hbox = QHBoxLayout()
        grid = QGridLayout()
        self.link = link
        self.setWindowTitle('游戏推荐系统 登录')

        self.username = QLineEdit()
        self.username.setEchoMode(QLineEdit.Normal)
        self.username.setPlaceholderText('Username')
        self.username.setToolTip('请输入你的帐号')

        self.password = QLineEdit()
        self.password.setEchoMode(QLineEdit.Password)
        self.password.setPlaceholderText('Password')
        self.password.setToolTip('请输入你的密码')

        self.btnLogin = QPushButton('登录')

        hbox.addWidget(self.username)
        hbox.addWidget(self.password)
        grid.addWidget(self.btnLogin)
        layout.addLayout(hbox, 0, 1)
        layout.addLayout(grid, 1, 1)
        self.btnLogin.clicked.connect(self.onClicked)

    def onClicked(self):
        username = self.username.text()
        password = self.password.text()
        if username == '' or password == '':
            return

        if username in user2hash.keys():
            self.btnLogin.setText('生成推荐中，此过程可能耗时一分钟……')
            self.setWindowTitle('生成推荐中……(~1min)')
            self.user = username
            self.link = Window(link=self)
            self.link.show()
            self.close()

app = QApplication(sys.argv)
window = QSplitter()
login = Login(link=window)
login.show()
sys.exit(app.exec_())


history = set()
user = input('user id: ')
print('user id hashed: {}'.format(user2hash[user]))
print('-------------------history-------------------------')
for line in open('../data/steam/ratings.tsv', encoding='utf-8').readlines():
    u = line.strip().split('\t')[0]
    i = line.strip().split('\t')[1]
    if u == user:
        print('{:10d} {}'.format(item2hash[i], i))
        history.add(item2hash[i])
print('----------------recommendation---------------------')

# data = [[user2hash[user], int(v), 0] for v in item2hash.values()]
# data = []
# for v in item2hash.values():
#     if v not in history:
#         data.append([user2hash[user], int(v), ])
# data = np.array(data)
model = RippleNet(args, n_entity, n_relation)
state_dict = torch.load('../ripplenet.pth')
model.load_state_dict(state_dict)
model.eval()

self.scores = []
data = np.array([[user2hash[user], 0, 0]])
feed_data = get_feed_dict(args, model, data, ripple_set, 0, 1)
for v in item2hash.values():
    # if v not in history:
        feed_data[0][0] = v
        return_dict = model(*feed_data)
        self.scores.append(return_dict['self.scores'].cpu().detach().item())
# while start < data.shape[0]:
#     return_dict = model(*)
#     self.scores.append(return_dict['self.scores'].cpu().detach().numpy().tolist())
print('prediction finished.')
self.scores = np.array(self.scores).reshape(-1)

hash2item = list(item2hash.keys())
self.scores_indices = (-self.scores).argsort()

writer = open('../recommendation_for_{}.tsv'.format(user), 'w', encoding='utf-8')
# for i in range(len(self.scores_indices)):
for i in range(10):
    # writer.write('{:10d}\t{}\t{}\n'.format(self.scores_indices[i], hash2item[self.scores_indices[i]], self.scores[self.scores_indices[i]]))
    print('{:10d} {}'.format(self.scores_indices[i], hash2item[self.scores_indices[i]]))
