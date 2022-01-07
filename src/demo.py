import sys
import argparse
import numpy as np
from data_loader import load_data
from predict import predict
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QSplitter, QWidget, QListWidget, QListWidgetItem, \
    QTableWidget, QTableWidgetItem, QApplication, QAbstractItemView, \
    QGridLayout, QHBoxLayout, QLineEdit, QPushButton
from PyQt5.QtCore import Qt

np.random.seed(555)  # 随机种子

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

data_info = load_data(args)  # 加载相关数据集


class History(QListWidget):
    def __init__(self, parent=None, link=None):
        super(History, self).__init__(parent)
        self.history = []
        self.user = link.user
        title = QListWidgetItem()
        title.setText('User History Below:\n')
        self.addItem(title)
        file = '../data/movie/ratings.dat'
        for line in open(file, encoding='utf-8').readlines():
            usr = line.strip().split('::')[0]
            movie = line.strip().split('::')[1]
            if int(usr) == self.user:
                self.history.append(movie2info[movie])
                child = QListWidgetItem()
                child.setText(movie2info[movie])
                self.addItem(child)


class Recommendation(QTableWidget):
    def __init__(self, parent=None, link=None):
        super(Recommendation, self).__init__(parent)
        self.user = link.user

        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setHorizontalScrollMode(QListWidget.ScrollPerPixel)
        self.setColumnCount(3)
        self.setRowCount(10)
        self.setHorizontalHeaderLabels(['movie_id recommended', 'movie_title recommended', 'movie_channel belonged'])

        top_k, movie_top_k = predict(args, data_info, self.user)  # 对传入的user进行推荐，返回top_k的电影序列

        i = 0
        for i in range(len(top_k)):
            item = QTableWidgetItem(str(top_k[i]))
            self.setItem(i, 0, item)

            for j in range(1, 3):
                item = QTableWidgetItem(movie_top_k[i][j-1])
                self.setItem(i, j, item)
            i += 1

        self.tail = i


class Window(QSplitter):
    def __init__(self, parent=None, link=None):
        super(Window, self).__init__(parent)
        self.setWindowTitle('Recommendation system on movies')
        self.user = link.user
        self.history = History(self, link=link)
        self.recommendation = Recommendation(self, link=link)
        self.addWidget(self.history)
        self.addWidget(self.recommendation)
        self.setOrientation(Qt.Horizontal)
        self.setSizes([400, 800])
        self.resize(1000, 500)

    def closeEvent(self, event):
        pass


class Login(QWidget):
    def __init__(self, parent=None, link=None):
        super(Login, self).__init__(parent)

        layout = QGridLayout(self)
        h_box = QHBoxLayout()
        grid = QGridLayout()
        self.link = link

        self.setWindowTitle('Login recommendation system.')

        self.username = QLineEdit()
        self.username.setEchoMode(QLineEdit.Normal)
        self.username.setPlaceholderText('Username')
        self.username.setToolTip('Input your username:')

        self.password = QLineEdit()
        self.password.setEchoMode(QLineEdit.Password)
        self.password.setPlaceholderText('Password')
        self.password.setToolTip('Input your password:')

        self.btnLogin = QPushButton('login now!')

        h_box.addWidget(self.username)
        h_box.addWidget(self.password)
        grid.addWidget(self.btnLogin)
        layout.addLayout(h_box, 0, 1)
        layout.addLayout(grid, 1, 1)
        self.btnLogin.clicked.connect(self.onClicked)

    def onClicked(self):
        username = self.username.text()
        password = self.password.text()
        if username == '' or password == '':
            return

        if (username, password) in user2id.keys():
            self.btnLogin.setText('Logged!')
            self.setWindowTitle('recommending...')

            self.user = user2id[username, password]
            self.link = Window(link=self)
            self.link.show()
            self.close()

        else:
            print('no such user, pls check and input again.')


user2id = dict()
for line in open('../data/movie/user2id.txt', encoding='utf-8').readlines():
    user = line.strip().split('::')[0]
    password = line.strip().split('::')[1]
    userid = line.strip().split('::')[2]
    user2id[user, password] = int(userid)

movie2info = dict()
for line in open('../data/movie/movies.dat', encoding='utf-8').readlines():
    movie = line.strip().split('::')[0]
    info = line.strip().split('::')[1]
    movie2info[movie] = info

app = QApplication(sys.argv)
window = QSplitter()
login = Login(link=window)
login.show()
sys.exit(app.exec_())
