import pickle
from utils import build_graph, Data, split_validation
import networkx as nx
import matplotlib.pyplot as plt

if __name__ == '__main__':
    test_data = pickle.load(open('../datasets/yoho1_64/train_300.txt', 'rb'))
    print(len(test_data[0]))
    print(len(test_data[1]))

    # test_data = Data(test_data, sub_graph=True, method='ggnn', shuffle=False)

    G = build_graph(test_data[0])
    nx.draw(G, with_labels=True)
    plt.show()

    # G = nx.Graph()  # 建立一个空的无向图G
    # G.add_node('a')  # 添加一个节点1
    # G.add_nodes_from(['b', 'c', 'd', 'e'])  # 加点集合
    # G.add_cycle(['f', 'g', 'h', 'j'])  # 加环
    # H = nx.path_graph(10)  # 返回由10个节点挨个连接的无向图，所以有9条边
    # G.add_nodes_from(H)  # 创建一个子图H加入G
    # G.add_node(H)  # 直接将图作为节点
    #
    # nx.draw(G, with_labels=True)
    # plt.show()
