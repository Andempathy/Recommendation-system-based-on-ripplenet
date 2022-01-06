import numpy as np
import random
import torch
import os
from model import RippleNet


def predict(args, data_info, usr):
    n_entity = data_info[3]
    n_relation = data_info[4]
    ripple_set = data_info[5]
    movie_index_item2entity = get_movie_index()
    top_k = []
    movie_top_k = []

    model = RippleNet(args, n_entity, n_relation)  # 初始化模型

    if args.use_cuda:
        model.cuda()

    model_save_path = os.path.join(args.model_dir, 'ripplenet.pt')

    if os.path.exists(model_save_path):
        loaded_paras = torch.load(model_save_path)
        model.load_state_dict(loaded_paras)  # 加载模型参数再次初始化

        model.eval()

        user_id, user_id_old = get_user_id(usr)
        # 获取一个userid，准备为其推荐电影
        # user_id = 2333
        # user_id_old = 2233
        items = get_items(user_id, movie_index_item2entity)
        scores = get_scores(args, model, items, ripple_set)
        top_k = get_top_k(items, scores, k=args.k)
        movie_top_k = get_movie_info(top_k, user_id_old)

        model.train()
    else:
        print('No model saved, please train firstly.')
    return top_k, movie_top_k


# 获取一个userid
def get_user_id(usr=None):
    user_index_old2new = dict()
    file = '../data/movie/user_index_old2new.txt'
    for line in open(file, encoding='utf-8').readlines():
        user_index_old = line.strip().split('\t')[0]
        user_index_new = line.strip().split('\t')[1]
        user_index_old2new[user_index_old] = user_index_new

    if usr is not None:
        user_id_old = str(usr)
    else:
        user_id_old = random.choice(list(user_index_old2new))
        print('Now, you randomly get a userid：%s.\n' % user_id_old)

    user_id_new = user_index_old2new[user_id_old]
    return int(user_id_new), int(user_id_old)


# 获取movie_id的对应关系
def get_movie_index():
    movie_index_item2entity = dict()
    file = '../data/movie/item_index2entity_id_rehashed.txt'

    for line in open(file, encoding='utf-8').readlines():
        array = line.strip().split('\t')
        movie_index_item2entity[array[0]] = array[1]

    return movie_index_item2entity


# 基于movie_id返回movie的简要标识信息，作为最终的推荐结果
def get_movie_info(item_list, user_id):
    file = '../data/movie/movies.dat'
    movie_index_id2info = {}

    for line in open(file, encoding='utf-8').readlines():
        array = line.strip().split('::')
        movie_index_id2info[int(array[0])] = list([array[1], array[2]])

    movie_list = [movie_index_id2info[i] for i in item_list]

    print('Now, you get the movies recommend for the user with id:%d.' % user_id)
    for i in range(len(movie_list)):
        print('%d:\t%s\t%s' % (i + 1, movie_list[i][0], movie_list[i][1]))
    return movie_list


# 为userid构造相应的三元组序列
def get_items(user_id, movie_index_item2entity):
    n_item = 0
    n_rating = 0
    n_movies = 3951

    item_set = set(range(1, n_movies))
    entity_set = set()

    file = '../data/movie/ratings.dat'
    for line in open(file, encoding='utf-8').readlines():
        array = line.strip().split('::')
        if int(array[0]) == user_id:  # 查中
            item_set.discard(int(array[1]))  # 从备选集中剔除所查中的id
            n_rating += 1

    for i in item_set:
        if str(i) in movie_index_item2entity.keys():
            entity_set.add(int(movie_index_item2entity[str(i)]))  # 向集合中添加转换后的id
            n_item += 1

    items = np.empty((n_item, 3), int)
    # 三元组序列中，首先是userid，有唯一值，是前一步确定的
    items[:, 0] = user_id
    # 其次是item_id，确切的说是item_id转换后得到的entity_id，而且是排除了已有本用户评分的id
    items[:, 1] = list(entity_set)
    # 最后是label，全部置0
    items[:, 2] = 0
    return items


# 喂入多跳结果集
def get_feed_dict(args, data, ripple_set):
    # 和train.py的同名函数类似，区别在于此处仅对单个item做处理
    items = torch.LongTensor(data[:, 1])
    labels = torch.LongTensor(data[:, 2])
    memories_h, memories_r, memories_t = [], [], []

    for i in range(args.n_hop):
        memories_h.append(torch.LongTensor([ripple_set[user][i][0] for user in data[:, 0]]))
        memories_r.append(torch.LongTensor([ripple_set[user][i][1] for user in data[:, 0]]))
        memories_t.append(torch.LongTensor([ripple_set[user][i][2] for user in data[:, 0]]))

    if args.use_cuda:
        items = items.cuda()
        labels = labels.cuda()
        memories_h = list(map(lambda x: x.cuda(), memories_h))
        memories_r = list(map(lambda x: x.cuda(), memories_r))
        memories_t = list(map(lambda x: x.cuda(), memories_t))
    return items, labels, memories_h, memories_r, memories_t


# 从模型返回中获取分数
def get_scores(args, model, items, ripple_set):  # 获取items的得分
    return_dict = model.forward(*get_feed_dict(args, items, ripple_set))
    scores = return_dict["scores"].detach().cpu().numpy()
    return scores


# 将返回的得分进行排序，选择最高得分的电影序列
def get_top_k(items, scores, k):
    index = scores.argsort()[::-1]
    index_k = index[0:k]
    top_k = items[index_k, 1]
    return list(top_k)
