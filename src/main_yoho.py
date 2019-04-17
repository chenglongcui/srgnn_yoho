from __future__ import division
import numpy as np
from model import *
from utils import build_graph, Data, split_validation
import pickle
import argparse
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# logger.setLevel(level=logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler = logging.FileHandler("log.txt")
handler.setLevel(logging.INFO)
#
handler.setFormatter(formatter)
logger.addHandler(handler)
# Check Model Dir
item_pred = {}
DATA_PATH = ''
model_dir = DATA_PATH + 'model/ckpt'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
parser.add_argument('--train', type=int, default=1)
parser.add_argument('--pred_cnt', type=int, default=20)
parser.add_argument('--method', type=str, default='ggnn', help='ggnn/gat/gcn')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--epoch', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--nonhybrid', action='store_true', help='global preference')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
opt = parser.parse_args()
logger.info(opt)
is_train = opt.train

logger.info("Start Loading training data")
train_data = pickle.load(open('../datasets/' + opt.dataset + '/train_1000.txt', 'rb'))

logger.info("Start Loading valid data")
valid_data = pickle.load(open('../datasets/' + opt.dataset + '/valid.txt', 'rb'))
logger.info("Start Loading test data")
test_data = pickle.load(open('../datasets/' + opt.dataset + '/test.txt', 'rb'))
# print(test_data['88'])
item_dict = pickle.load(open('../datasets/item_dict.txt', 'rb'))
val_idx = pickle.load(open('../datasets/val_idx.txt', 'rb'))
test_item_cnt = pickle.load(open('../datasets/test_item_cnt.txt', 'rb'))

# print(val_idx)
# print(len(test_item_cnt))
n_node = len(item_dict) + 1

print(len(test_item_cnt))
train_data = Data(train_data, sub_graph=True, method=opt.method, shuffle=True)
#
# print(len(train_data.inputs))
# valid_data = Data(valid_data, sub_graph=True, method=opt.method, shuffle=False)
# print(opt.lr_dc_step * len(train_data.inputs) / opt.batchSize)
# model = GGNN(hidden_size=opt.hiddenSize, out_size=opt.hiddenSize, batch_size=opt.batchSize, n_node=n_node,
#              lr=opt.lr, l2=opt.l2, step=opt.step, decay=opt.lr_dc_step * len(train_data.inputs) / opt.batchSize,
#              lr_dc=opt.lr_dc,
#              nonhybrid=opt.nonhybrid)
# # print(opt)
# best_result = [0, 0]
# best_epoch = [0, 0]
# logger.info("Loading data Finished")
# if is_train:
#     for epoch in range(opt.epoch):
#         # print('epoch: ', epoch, '===========================================')
#         logger.info('epoch: %d ===========================================' % epoch)
#         slices = train_data.generate_batch(model.batch_size)
#         print(slices)
#         # print(len(slices))
#         fetches = [model.opt, model.loss_train, model.global_step, model.learning_rate]
#         # print('start training: ', datetime.datetime.now())
#         logger.info("Start Training")
#         loss_ = []
#
#         # Train Step
#         if is_train:
#             for i, j in zip(slices, np.arange(len(slices))):
#                 adj_in, adj_out, alias, item, mask, targets = train_data.get_slice(i)
#                 # print("adj_in", adj_in[0].shape)
#                 # print("adj_out", adj_out[0].shape)
#                 _, loss, step, lr = model.train(fetches, targets, item, adj_in, adj_out, alias, mask)
#                 loss_.append(loss)
#                 if step % 20 == 0:
#                     logger.info("step: %d, training loss: %.4f, learning rate: %.8f" % (step, loss, lr))
#             model.save_model(model_dir)
#             loss = np.mean(loss_)
#
#     # Predict Step
#     # if (epoch % 3 == 0) & (epoch != 0):
#     # print("inputs", type(valid_data.inputs[0]))
#     # print("targets", valid_data.targets[0])
# else:
#     model.restore_model(model_dir)
#     out_file_skn = open(DATA_PATH + 'pred_skn.txt', "w")
#     slices = valid_data.generate_batch(model.batch_size)
#     # print("slices")
#     # print(slices)
#
#     logger.info('start predicting')
#     hit, mrr, test_loss_ = [], [], []
#     # hit, test_loss_ = [], []
#
#     for i, j in zip(slices, np.arange(len(slices))):
#         adj_in, adj_out, alias, item, mask, targets = valid_data.get_slice(i)
#
#         index_topk = model.pred([model.arg_sort], item, adj_in, adj_out, alias, mask)
#         index_topk = index_topk[0]
#         for idx in range(len(i)):
#             sess_id = val_idx[i[idx]]
#             skn_index_list = index_topk[idx]
#             for skn_rank in range(len(skn_index_list)):
#                 # print(sess_id)
#                 output_str = "%s,%s,%d\n" % (sess_id, item_dict[skn_index_list[skn_rank] + 1], skn_rank + 1)
#                 # print(output_str)
#                 # out_file_skn.write(output_str)
#                 item = item_dict[skn_index_list[skn_rank] + 1]
#                 if sess_id in item_pred:
#                     item_pred[sess_id] += [item]
#                 else:
#                     item_pred[sess_id] = [item]
#     print(len(item_pred))
#     # test_loss_.append(test_loss)
#
#     # evaluation
#     sess_hit = {}
#     hit = []
#     for sess_id in test_data:
#         # print(sess_id)
#         # print(test_data[sess_id])
#         sess_click_test = test_data[sess_id]
#         hit_cnt = 0
#         if sess_id in item_pred:
#             for item in item_pred[sess_id]:
#                 if item in sess_click_test:
#                     hit_cnt += 1
#             sess_hit[sess_id] = hit_cnt
#
#     for sess in sess_hit:
#         hit_rate = sess_hit[sess] / test_item_cnt[sess]
#         hit.append(hit_rate)
#     logger.info("lenth of evaluation: %d" % (len(sess_hit)))
#     logger.info(np.mean(np.array(hit)) * 100)
#     # print(len(hit))
#     # hit = np.mean(mrr) * 100
#     # print(hit)
#
#     # for score, target in zip(index, targets):
# #     for score, target in zip(index_topk, targets):
# #         hit.append(np.isin(target - 1, score))
# #         if len(np.where(score == target - 1)[0]) == 0:
# #             mrr.append(0)
# #         else:
# #             mrr.append(1 / (20 - np.where(score == target - 1)[0][0]))
# # hit = np.mean(hit) * 100
# # mrr = np.mean(mrr) * 100
# # test_loss = np.mean(test_loss_)
# # if hit >= best_result[0]:
# #     best_result[0] = hit
# #     best_epoch[0] = epoch
# # if mrr >= best_result[1]:
# #     best_result[1] = mrr
# #     best_epoch[1] = epoch
# # # print('train_loss:\t%.4f\ttest_loss:\t%4f\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d' %
# # #       (loss, test_loss, best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
# # logger.info('train_loss:\t%.4f\ttest_loss:\t%4f\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d' %
# #             (loss, test_loss, best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
