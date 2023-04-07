import datetime
import logging
import random
import torch
from scipy.stats import norm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from torch.optim.adam import Adam
import time
from torch.utils.data import Dataset, DataLoader
import pickle as pkl
from tqdm import tqdm
import scipy.stats as st

SEED            = 0
EPOCH           = 500
LR              = 0.001
WEIGHT_DECAY    = 1e-6
BATCH_SIZE      = 128
DEVICE          = 'cuda:5'
K               = 5 # top-K
N_l             = 65
CONFIDENCE      = 0.95
TRAIN_PATH      = 'your path'
TEST_PATH       = 'your path'
LOG_PATH        = 'your path'
LABEL_RATE      = 0.008



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
file_handler = logging.FileHandler(LOG_PATH+'log.txt')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
exp_time = datetime.datetime.now()
logger.info("\n\n\n\n-----------------------------Start-----------------------------------------")
logger.info(exp_time)



class MultimodalLSTM(nn.Module):
    def __init__(self, T_l, N_l):
        super(MultimodalLSTM, self).__init__()
        self.T_l = T_l
        self.N_l = N_l
        self.hidden_size=N_l
        self.num_layers=3
        self.lstm_d1 = nn.LSTM(input_size=N_l, hidden_size=N_l, num_layers=1, batch_first=True)
        self.lstm_d2 = nn.LSTM(input_size=1, hidden_size=1, num_layers=1, batch_first=True)
        self.lstm_middle = nn.LSTM(input_size=N_l+1, hidden_size=N_l+1, num_layers=self.num_layers, batch_first=True)
        self.lstm_o1 = nn.LSTM(input_size=N_l, hidden_size=N_l, num_layers=self.num_layers, batch_first=True)
        self.lstm_o2 = nn.LSTM(input_size=1, hidden_size=1, num_layers=self.num_layers, batch_first=True)
        # self.softmax_d1 = nn.Softmax(dim=2)，torch的交叉熵自带softmax

    def forward(self, d1, d2):
        """
        一个batch大小的d1和d2作为输入，输出预测差分出下一”时刻”的d1和d2
        :param d1: (B, T_l, N_l)
        :param d2: (B, T_l, 1)
        :return: d1_predict(B, T_l, N_l), d2_predict(B, T_l, 1)
        """
        B, T_l, N_l = d1.shape
        B, T_l, _ = d2.shape

        # 左下那个lstm
        h0 = Variable(torch.zeros(1, B, N_l)).to(DEVICE) # h0的shape是(层数,batch大小,hidden_size)
        c0 = Variable(torch.zeros(1, B, N_l)).to(DEVICE) # c0的shape是(层数,batch大小,hidden_size)
        output1, (h1, c1) = self.lstm_d1(d1, (h0, c0))

        # 右下那个lstm
        h0 = Variable(torch.zeros(1, B, 1)).to(DEVICE) # h0的shape是(层数,batch大小,hidden_size)
        c0 = Variable(torch.zeros(1, B, 1)).to(DEVICE) # c0的shape是(层数,batch大小,hidden_size)
        output2, (h2, c2) = self.lstm_d2(d2, (h0, c0))

        # 还需要把h和c传给中间的那个lstm
        h3 = torch.cat((h1, h2), 2)
        c3 = torch.cat((c1, c2), 2)
        # 为什么这里要repeat，文档不是说输出的shape也是(层数,batch大小,hidden_size)吗？
        # https://pytorch.org/docs/master/generated/torch.nn.LSTM.html#torch.nn.LSTM
        h3 = h3.repeat(self.num_layers, 1, 1)
        c3 = c3.repeat(self.num_layers, 1, 1)
        output3 = torch.cat((output1, output2), 2)

        # 中间的那个lstm
        output, (h4, c4) = self.lstm_middle(output3, (h3, c3))

        # output还要分别经过各自的lstm
        predict_d1, (_, _) = self.lstm_o1(output[:, :, :-1])
        predict_d2, (_, _) = self.lstm_o2(output[:, :, -1].unsqueeze(-1))

        # self.softmax_d1 = nn.Softmax(dim=2)，torch的交叉熵自带softmax
        return predict_d1, predict_d2


# 网络的Trainer
class MultimodalLSTMTrainer(object):
    def __init__(self):
        # Results
        self.support = None
        self.f1_score = None
        self.recall = None
        self.precision = None

    def loss_function(self,
                      predict_d1: torch.Tensor,
                      predict_d2: torch.Tensor,
                      d1: torch.Tensor,
                      d2: torch.Tensor):
        B, T_l, N_l = d1.shape
        B, T_l, _ = d2.shape
        shifted_d1 = d1.roll(-1, 1)
        shifted_d1 = shifted_d1[:, :-1] # 去掉最后一项
        shifted_d2 = d2.roll(-1, 1)
        shifted_d2 = shifted_d2[:, :-1] # 去掉最后一项
        cel = nn.CrossEntropyLoss()
        CEL = cel(
            predict_d1[:, :-1].reshape(-1, N_l),
            torch.topk(shifted_d1.reshape(-1, N_l), 1)[1].squeeze(1)
        )
        mse = nn.MSELoss()
        MSE = mse(predict_d2[:, :-1].flatten(), shifted_d2.flatten())
        total_loss = CEL + MSE / (MSE / CEL).detach() # 均衡一下，参考的是https://www.zhihu.com/question/375794498/answer/1052779937
        # total_loss = CEL + MSE
        # total_loss = CEL
        return total_loss, CEL, MSE
        

    def train(self, train_dataloader, validate_dataloader, net: MultimodalLSTM, optimizer) -> MultimodalLSTM:
        start_time = time.time()
        for epoch in range(EPOCH):
            net.train()
            loss_list   = []
            CEL_list    = []
            MSE_list    = []

            epoch_start_time = time.time()
            for data in train_dataloader:
                train_d1, train_d2, label = data

                train_d1 = train_d1.float().to(DEVICE)
                train_d2 = train_d2.float().to(DEVICE)

                predict_d1, predict_d2 = net(train_d1, train_d2)

                loss, CEL, MSE = self.loss_function(predict_d1,
                                                    predict_d2,
                                                    train_d1,
                                                    train_d2)
                # BP
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_list.append(loss.item())
                CEL_list.append(CEL.item())
                MSE_list.append(MSE.item())

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info('  Epoch {}/{}\t Time:{:.3f}\t avg_loss:{:.8f} \tavg_cel:{:.8f} \tavg_mse:{:.8f} \te^p:{:.8f}'
                        .format(epoch + 1,
                                EPOCH,                                            # epoch数
                                epoch_train_time,                                 # 本epoch耗时
                                np.mean(loss_list),                               # epoch平均总损失
                                np.mean(CEL_list),                                # epoch平均cel
                                np.mean(MSE_list),                                # epoch平均MSE
                                np.exp(-1*np.mean(loss_list)),                    # epoch平均正类概率，根据CEL的公式可以知道这个值
                                ))
            if epoch % 30 == 0 and epoch > 0:
                self.test(test_dataloader, net)
        train_time = time.time() - start_time
        logger.info('training time: %.3f' % train_time)
        logger.info('Finished training.')
        return net

    def test(self, test_dataloader, net: MultimodalLSTM):
        logger.info('Testing...')
        net.eval()
        op_error = []
        for i in range(0, N_l):
            op_error.append([])
        topk_sad_list = []
        real_sad_list = []
        square_error_list = []
        label_list = []
        with torch.no_grad():
            # 先利用训练集构建高斯模型
            for data in tqdm(train_dataloader):
                train_d1, train_d2, label = data
                train_d1, train_d2 = train_d1.float().to(DEVICE), train_d2.float().to(DEVICE)
                train_predict_d1, train_predict_d2 = net(train_d1, train_d2)
                shifted_train_d1 = train_d1.roll(-1, 1) # (B, T_l, N_l)
                shifted_train_d1 = torch.argmax(shifted_train_d1, 2).cpu().numpy() # (B, T_l)
                shifted_train_d2 = train_d2.roll(-1, 1) # (B, T_l)
                square_error = (train_predict_d2 - shifted_train_d2)**2 # (B, T_l)
                B, T_l = shifted_train_d1.shape
                shifted_train_d1 = shifted_train_d1.tolist() # numpy太慢了，要转list
                square_error = square_error.tolist() # numpy太慢了，要转list
                for i in range(B):
                    for j in range(T_l):
                        op_error[shifted_train_d1[i][j]].append(square_error[i][j])
            for data in tqdm(test_dataloader):
                test_d1, test_d2, label = data
                test_d1, test_d2, label = test_d1.float().to(DEVICE), test_d2.float().to(DEVICE), label.to(DEVICE)
                test_predict_d1, test_predict_d2 = net(test_d1, test_d2)
                test_predict_d1 = torch.softmax(test_predict_d1, dim=2)
                # 对于SAD需要获取topk, (B, T_l, N_l) -> (B, T_l, k)
                values, indices = torch.topk(test_predict_d1, K, 2) # 应该使用indices
                topk_sad_list.append(indices.cpu().numpy())
                shifted_test_d1 = test_d1.roll(-1, 1)
                real_sad_list.append(torch.argmax(shifted_test_d1, 2).cpu().numpy()) # (B, T_l)
                # 对于RTAD需要根据来判断方差是否在预测出的operation所对应的置信区间内，先记住，后面一起算
                shifted_test_d2 = test_d2.roll(-1, 1)
                square_error = (test_predict_d2 - shifted_test_d2)**2
                square_error_list.append(square_error.cpu().numpy().squeeze()) # (B, T_l)
                # label
                label_list.extend(label.cpu().tolist())
        topk_sad_np = np.vstack(topk_sad_list) # (N, T_l, k)
        real_sad_np = np.vstack(real_sad_list) # (N, T_l)
        square_error_np = np.vstack(square_error_list) # (N, T_l)
        # 先判断SAD异常
        real_sad_np = np.expand_dims(real_sad_np, 2) # (N, T_l, 1)
        SAD_normal = np.any(real_sad_np[:, :-1, :] == topk_sad_np[:, :-1, :], axis=2) # (N, T_l-1)，因为最后一个不需要正确
        SAD_normal = np.all(SAD_normal, axis=1) # (N,), True表示SAD正常，False表示SAD异常
        # 判断RTAD异常
        ## 先计算训练集得出的各个operation的置信区间
        op_interval = []
        for i in range(0, N_l):
            current_se = op_error[i]
            op_interval.append(st.t.interval(CONFIDENCE, len(current_se)-1, loc=np.mean(current_se), scale=st.sem(current_se)))
        N, T_l = square_error_np.shape
        square_error_np = square_error_np.tolist() # np太慢，转list
        topk_sad_np = topk_sad_np.tolist() # np太慢，转list
        RTAD_normal_list = []
        for i in tqdm(range(N)):
            isNormal = True
            for j in range(T_l):
                current_se = square_error_np[i][j]
                current_op = topk_sad_np[i][j]# 预测出来有k个可能的op，怎么知道取哪个置信区间? 直接认为只有当所有置信区间都不符合的情况下才有异常
                wrong_cnt = 0
                for k in range(K):
                    if op_interval[current_op[k]][1]<current_se: # 方差是个正数，只需判断右端
                        wrong_cnt += 1
                if wrong_cnt == K:
                    isNormal = False
                    break
            RTAD_normal_list.append(isNormal)
        RTAD_normal = np.array(RTAD_normal_list)
        # 最终结果
        predict = np.invert(np.bitwise_and(SAD_normal, RTAD_normal)).astype(int)
        # predict = np.invert(SAD_normal).astype(int) # 先只用SAD看看
        # 评价指标
        labels = np.array(label_list)
        self.precision, self.recall, self.f1_score, self.support = precision_recall_fscore_support(labels, predict)
        acc = accuracy_score(labels, predict)
        logger.info('Test set average precision:{:.2f}%, recall:{:2f}%, F1: {:.2f}%'.format(100.*np.mean(self.precision), 100.*np.mean(self.recall), 100.*np.mean(self.f1_score)))
        logger.info('Test set class 0 precision:{:.2f}%, recall:{:2f}%, F1: {:.2f}%'.format(100.*self.precision[0], 100.*self.recall[0], 100.*self.f1_score[0]))
        logger.info('Test set class 1 precision:{:.2f}%, recall:{:2f}%, F1: {:.2f}%'.format(100.*self.precision[1], 100.*self.recall[1], 100.*self.f1_score[1]))
        logger.info('Test set accuracy:{:.2f}%'.format(100.*acc))
        logger.info(f'struct_abnormal_cnt: {len(np.where(SAD_normal==False)[0])},\nrt_abnormal_cnt: {len(np.where(RTAD_normal==False)[0])}')
        logger.info('trace num in test set: {}'.format(len(labels)))
        return  100.*self.precision[0], 100.*self.recall[0], 100.*self.f1_score[0], \
                100.*self.precision[1], 100.*self.recall[1], 100.*self.f1_score[1], \
                100.*acc


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, path, isTrain):
        super().__init__()
        if isTrain:
            sad   = np.load(path.format("SAD"))
            rtad  = np.load(path.format("RTAD"))
            n_LN  = (int)(np.floor(len(sad)*LABEL_RATE))
            normal_idx_perm = np.random.permutation(len(sad))
            idx_chosen = normal_idx_perm[:n_LN]
            self.sad    = sad[idx_chosen]
            self.rtad   = rtad[idx_chosen]
            self.label  = np.zeros([len(self.sad)])
        else:
            self.sad    = np.load(path.format("SAD"))
            self.rtad   = np.load(path.format("RTAD"))
            self.label  = np.load(path.format("label"))

    def __len__(self):
        return len(self.sad)

    def __getitem__(self, idx):
        return self.sad[idx], self.rtad[idx], self.label[idx]


if SEED != -1:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    logger.info('Set seed to %d.' % SEED)

train_dataset = MyDataset(TRAIN_PATH, True)
test_dataset = MyDataset(TEST_PATH, False)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

net = MultimodalLSTM(train_dataset.sad.shape[1], train_dataset.sad.shape[2]).to(DEVICE)
optimizer = Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
trainer = MultimodalLSTMTrainer()
net = trainer.train(train_dataloader, test_dataloader, net, optimizer)
NPR, NRE, NF1, APR, ARE, AF1, ACC = trainer.test(test_dataloader, net)