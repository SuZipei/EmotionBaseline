import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Permute(nn.Module):
    def __init__(self):
        super(Permute, self).__init__()

    def forward(self, x):
        return x.permute(0, 2, 1)


class LayerNorm(nn.Module):
    def __init__(self, hnshape):
        super(LayerNorm, self).__init__()
        self.activation = nn.LeakyReLU()
        self.g = nn.Parameter(torch.Tensor(hnshape[0], hnshape[1], hnshape[2]))
        self.b = nn.Parameter(torch.Tensor(hnshape[0], hnshape[1], hnshape[2]))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.g, 0.5)
        # nn.init.constant_(self.g, 0.2)
        # nn.init.xavier_normal_(self.g)
        # nn.init.xavier_uniform_(self.g)
        # nn.init.constant_(self.b, 0.1)
        nn.init.xavier_uniform_(self.b)
        # nn.init.xavier_normal_(self.b)

    def forward(self, ht):
        # ht((num_layers * num_directions, batch, hidden_size))
        ut = ht.mean(-1).unsqueeze(-1)
        dt = ((((ht - ut)**2).mean(-1))**1/2).unsqueeze(-1)
        yt = self.activation((self.g / dt) * (ht - ut) + self.b)
        return yt


class MMResLstmBlock(nn.Module):
    def __init__(self, indim, outdim, num_layer, batch_size, seq_len, wh):
        super(MMResLstmBlock, self).__init__()
        self.batch_size = batch_size
        self.in_dim = indim
        self.out_dim = outdim
        self.num_layer = num_layer
        self.seq_len = seq_len
        self.lstm = nn.LSTM(input_size=self.in_dim, hidden_size=self.out_dim, num_layers=self.num_layer,
                            batch_first=True, bidirectional=False)
        self.lstm.weight_hh_l0 = wh
        self.lstm.flatten_parameters()
        self.layernorm = LayerNorm((self.batch_size,self.seq_len, self.out_dim))
        self.dropout = nn.Dropout(0.1)
        self.h0 = torch.zeros((self.num_layer, self.batch_size, self.out_dim)).to(device)
        self.c0 = torch.zeros((self.num_layer, self.batch_size, self.out_dim)).to(device)
        self.save_h0 = torch.zeros((self.num_layer, self.batch_size, self.out_dim)).to(device)
        self.save_c0 = torch.zeros((self.num_layer, self.batch_size, self.out_dim)).to(device)

    def forward(self, xt):
        with torch.no_grad():
            self.h0 = self.save_h0.clone()
            self.c0 = self.save_c0.clone()
        out, (hn, cn) = self.lstm(xt, (self.h0, self.c0))
        self.save_h0 = hn
        self.save_c0 = cn
        out = self.layernorm(out)
        out = self.dropout(out)
        return out


torch.autograd.set_detect_anomaly(True)


class Net(nn.Module):
    def __init__(self, model_config):
        super(Net, self).__init__()
        self.dim_eeg = int(model_config['in_eeg'])
        self.dim_pps = int(model_config['in_pps'])
        self.out_dim = int(model_config['out_dim'])
        self.num_layer = int(model_config['num_layer'])
        self.batch_size = int(model_config['batch'])
        self.num_block = int(model_config['num_block'])
        self.seq_len = int(model_config['sample_feature_num'])
        self.num_node = int(model_config['num_node'])
        self.eeg_blocks = nn.ModuleList()
        self.pps_block = nn.ModuleList()
        self.per1 = Permute()
        self.whs = []
        for i in range(self.num_block):
            self.whs.append(nn.Parameter(torch.Tensor(4 * self.out_dim, self.out_dim)))
        for i in range(self.num_block):
            if i == 0:
                self.eeg_blocks.append(
                    MMResLstmBlock(self.dim_eeg, self.out_dim, self.num_layer, self.batch_size, self.seq_len, self.whs[i]))
                self.pps_block.append(
                    MMResLstmBlock(self.dim_pps, self.out_dim, self.num_layer, self.batch_size, self.seq_len, self.whs[i]))
            else:
                self.eeg_blocks.append(
                    MMResLstmBlock(self.out_dim, self.out_dim, self.num_layer, self.batch_size, self.seq_len, self.whs[i]))
                self.pps_block.append(
                    MMResLstmBlock(self.out_dim, self.out_dim, self.num_layer, self.batch_size, self.seq_len, self.whs[i]))
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(self.seq_len * 2 * self.out_dim, 2)
        # self.dense2 = nn.Linear(128, 2)
        # self.dense3 = nn.Linear(32, 2)
        self.act = nn.LeakyReLU()
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.num_block):
            nn.init.xavier_uniform_(self.whs[i])

    def forward(self, data):
        x_T = data.TS
        #改变输入形状为b,n,f
        x_T = x_T.view((-1, self.num_node, x_T.shape[-1]))
        x_T = self.per1(x_T)
        x_eeg = x_T[:, :, :32]
        x_pps = x_T[:, :, 32:]
        out_eeg = x_eeg
        out_pps = x_pps
        #分不同的block来共享权重
        for i in range(self.num_block):
            if i == 0:
                out_eeg = self.eeg_blocks[i](out_eeg)
                out_pps = self.pps_block[i](out_pps)
            else:
                last_eeg = out_eeg
                last_pps = out_pps
                out_eeg = self.eeg_blocks[i](out_eeg)
                out_pps = self.pps_block[i](out_pps)
                out_eeg += last_eeg
                out_pps += last_pps
        out = torch.cat([out_eeg, out_pps], dim=-1)
        out = self.flatten(out)
        out = self.act(self.dense1(out))
        # out = self.act(self.dense2(out))
        # out = self.dense3(out)
        return out
