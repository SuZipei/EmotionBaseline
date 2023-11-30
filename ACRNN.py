import torch as t
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, model_config):
        super(Net, self).__init__()
        # 读取模型配置
        self.time_sample_rate = int(model_config['time_sample_rate'])  # 输入张量的时间轴维数
        self.W_hidden_dim = int(model_config['W_hidden_dim'])  # channel-wise注意力的编码中间维度
        self.cnn_kernel = int(model_config['cnn_kernel'])  # CNN卷积核个数
        self.cnn_kernel_width = int(model_config['cnn_kernel_width'])  # CNN卷积核宽度
        self.MaxPool2d_kernel_width = int(model_config['MaxPool2d_kernel_width'])  # 池化层核宽度
        self.LSTM_num = int(model_config['LSTM_num'])  #LSTM隐藏单元数
        self.dropout_rate = float(model_config['dropout_rate']) #分类层dropout比率
        # 注册层
        self.Dropout = nn.Dropout(self.dropout_rate)
        self.BatchNorm1d = nn.BatchNorm1d(64)
        self.BatchNorm2d_1 = nn.BatchNorm2d(32)
        self.BatchNorm2d_2 = nn.BatchNorm2d(self.cnn_kernel)
        self.W1 = nn.Linear(32, self.W_hidden_dim)  # 降维FC
        self.W2 = nn.Linear(self.W_hidden_dim, 32)  # 降维FC
        self.Tanh = nn.Tanh()
        self.Softmax = nn.Softmax(dim=1)
        self.CNN = nn.Conv2d(1, self.cnn_kernel, (32, self.cnn_kernel_width))
        self.ELU = nn.ELU()
        # self.MaxPool2d = nn.MaxPool2d(kernel_size=(1, 75), stride=(10, 10))
        self.MaxPool2d = nn.MaxPool2d(kernel_size=(1, self.MaxPool2d_kernel_width))
        self.LSTM = nn.LSTM(self.LSTM_num, 64, 2, batch_first=True)
        self.Wq = nn.Linear(64, 64)
        self.Wk = nn.Linear(64, 64)
        self.Wv = nn.Linear(64, 64)
        self.W3 = nn.Linear(64, 16)
        self.relu = nn.ReLU()
        self.W4 = nn.Linear(16, 2)

    def forward(self, data):
        data = data.TS.view(-1, 32, self.time_sample_rate)
        # data shape is (none, 32, 128)
        batch_size = data.shape[0]
        ## channel-wise attention
        s_ave = t.mean(data, dim=-1)
        s_tmp = self.relu(self.W1(s_ave))
        c_imp = self.Softmax(self.W2(s_tmp))
        ## grant importance to each channel
        # c_imp shape is (none, 32)
        # data shape is (none, 32, 128)
        c_imp = c_imp.unsqueeze(-1).expand(-1, 32, self.time_sample_rate)
        X = data * c_imp
        X = X.unsqueeze(-1)
        X = self.BatchNorm2d_1(X)
        ## cnn and pooling
        X = X.permute(0, 3, 1, 2)
        X = self.relu(self.CNN(X))
        X = self.MaxPool2d(X)
        X = self.BatchNorm2d_2(X)
        ## LSTM
        X = X.reshape(batch_size, 1, -1)
        X, LSTM_tmp = self.LSTM(X)
        X = X.squeeze(dim=1)
        X = self.BatchNorm1d(X)
        # self-attention
        q = self.Wq(X)
        k = self.Wk(X)
        v = self.Wv(X)
        att = self.Softmax(q * k) / 8
        X = att * v
        # classification
        X = self.Dropout(X)
        X = self.relu(self.W3(X))
        X = self.W4(X)
        return X


if __name__ == '__main__':
    model_config = {}
    model_config['W_hidden_dim'] = 10
    model_config['cnn_kernel'] = 40
    model_config['cnn_kernel_width'] = 45

    net = Net(model_config)

    input = t.rand(30, 32, 384)
    a = net(input)
    a = 1
