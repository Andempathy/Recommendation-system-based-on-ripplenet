import numpy as np


# 对齐item\entity的id
def read_item_index_to_entity_id_file():
    file = '../data/movie/item_index2entity_id_rehashed.txt'
    print('reading item index to entity id file: ' + file + ' ...')
    i = 0
    for line in open(file, encoding='utf-8').readlines():
        item_index = line.strip().split('\t')[0]
        satori_id = line.strip().split('\t')[1]
        item_index_old2new[item_index] = i
        # dict({'item_index':i})
        entity_id2index[satori_id] = i
        # dict({'satori_id':i})
        i += 1


# 转换评分数据，得到rating_final.txt：userid  itemid  rating
def convert_rating():
    file = '../data/movie/ratings.dat'

    print('reading rating file ...')
    item_set = set(item_index_old2new.values())
    # set(i,...)
    user_pos_ratings = dict()  # 正例
    user_neg_ratings = dict()  # 负例

    for line in open(file, encoding='utf-8').readlines()[1:]:
        array = line.strip().split(SEP)  # 分割
        item_index_old = array[1]
        # str(item_index)
        if item_index_old not in item_index_old2new:  # 没有对齐字段，跳过
            continue
        item_index = item_index_old2new[item_index_old]  # 转换index,old2new的一次更新
        # str:(i)
        user_index_old = int(array[0])
        # int(user_index)
        rating = float(array[2])
        # float(rating)
        if rating >= THRESHOLD:  # 以thershold判别正例与否
            if user_index_old not in user_pos_ratings:
                user_pos_ratings[user_index_old] = set()  # 初始化相应集合
            user_pos_ratings[user_index_old].add(item_index)  # 扩张这个集合
            # dict({user_index_old:set({item_index,...}),...})
        else:
            if user_index_old not in user_neg_ratings:
                user_neg_ratings[user_index_old] = set()
            user_neg_ratings[user_index_old].add(item_index)

    print('converting rating file ...')
    writer = open('../data/movie/ratings_final.txt', 'w', encoding='utf-8')
    user_cnt = 0

    for user_index_old, pos_item_set in user_pos_ratings.items():
        if user_index_old not in user_index_old2new:
            user_index_old2new[user_index_old] = user_cnt
            user_cnt += 1
        user_index = user_index_old2new[user_index_old]  # 更新转换user_index

        for item in pos_item_set:
            writer.write('%d\t%d\t1\n' % (user_index, item))
            # user_index    item    1
        unwatched_set = item_set - pos_item_set  # 差集

        # 构造用于后续训练、验证、测试的负例：
        # 区别于正例从pos_item_set中write，负例是基于差集构造出的
        if user_index_old in user_neg_ratings:
            unwatched_set -= user_neg_ratings[user_index_old]
            # 排除在原有负例中出现过的用户记录
        # 使用排除后的差集，随机构造与正例等量的负例
        for item in np.random.choice(list(unwatched_set), size=len(pos_item_set), replace=False):
            writer.write('%d\t%d\t0\n' % (user_index, item))

    writer.close()

    writer_too = open('../data/movie/user_index_old2new.txt', 'w', encoding='utf-8')
    for user_index_old, user_index_new in user_index_old2new.items():
        writer_too.write('%d\t%d\n' % (user_index_old, user_index_new))
    writer_too.close()

    print('number of users: %d' % user_cnt)
    print('number of items: %d' % len(item_set))


# 转换kg数据，得到kg_final.txt：h   r   t
def convert_kg():
    print('converting kg file ...')

    entity_cnt = len(entity_id2index)
    relation_cnt = 0

    writer = open('../data/movie/kg_final.txt', 'w', encoding='utf-8')

    files = [open('../data/movie/kg_part1_rehashed.txt', encoding='utf-8'),
             open('../data/movie/kg_part2_rehashed.txt', encoding='utf-8')]

    for file in files:
        for line in file:
            array = line.strip().split('\t')
            # 获取三元组
            head_old = array[0]
            relation_old = array[1]
            tail_old = array[2]

            # 转换为统一id
            if head_old not in entity_id2index:
                entity_id2index[head_old] = entity_cnt
                entity_cnt += 1
            head = entity_id2index[head_old]

            if tail_old not in entity_id2index:
                entity_id2index[tail_old] = entity_cnt
                entity_cnt += 1
            tail = entity_id2index[tail_old]

            if relation_old not in relation_id2index:
                relation_id2index[relation_old] = relation_cnt
                relation_cnt += 1
            relation = relation_id2index[relation_old]

            writer.write('%d\t%d\t%d\n' % (head, relation, tail))

    writer.close()
    print('number of entities (containing items): %d' % entity_cnt)
    print('number of relations: %d' % relation_cnt)


if __name__ == '__main__':
    # 处理原始数据集,得到便于利用的kg_final.txt、rating.txt文件

    SEP = '::'  # 分隔符
    THRESHOLD = 4  # 正负例临界值

    entity_id2index = dict()
    relation_id2index = dict()
    item_index_old2new = dict()
    user_index_old2new = dict()

    read_item_index_to_entity_id_file()
    convert_rating()
    convert_kg()

    print('done')
