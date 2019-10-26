import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.autograd import Variable
from torch.nn.init import xavier_normal_


class SubNet(nn.Module):
    def __init__(self, in_size, hidden_size, dropout):
        super(SubNet, self).__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        normed = self.norm(x)
        dropped = self.drop(normed)
        y_1 = functional.relu(self.linear_1(dropped))
        y_2 = functional.relu(self.linear_2(y_1))
        y_3 = functional.relu(self.linear_3(y_2))
        return y_3


class TextSubNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        super(TextSubNet, self).__init__()
        self.rnn = nn.LSTM(in_size, hidden_size,
                           num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        _, final_states = self.rnn(x)
        h = self.dropout(final_states[0].squeeze())
        y_1 = self.linear_1(h)
        return y_1


# initialize parameters for scanning window
def init_window_params(rank, hidden_dim, out_dim):
    win_factor = torch.nn.Parameter(torch.Tensor(rank, hidden_dim, out_dim))
    win_weight = torch.nn.Parameter(torch.Tensor(1, rank))
    win_bias = torch.nn.Parameter(torch.Tensor(1, out_dim))
    win_factor = xavier_normal_(win_factor)
    win_weight = xavier_normal_(win_weight)
    win_bias.data.fill_(0)
    return win_factor, win_weight, win_bias


class FNL2(nn.Module):
    """
    HPFN model two-layer no time
    """
    def ptp_layer(self, data_dict, data_type, layer_idx):
        const_bias = self.const_bias
        batch_size = data_dict['0'].shape[0]

        poly_order = self.poly_order[layer_idx]
        poly_norm = self.poly_norm
        euclid_norm = self.euclid_norm
        output_dim = self.inter_node_out_dims[layer_idx]

        modal_len = self.modal_lens[layer_idx]
        modal_win = self.modal_wins[layer_idx]

        win_factor = None
        win_weight = None
        win_bias = None
        if layer_idx == 0:
            win_factor = self.l1_win_factor
            win_weight = self.l1_win_weight
            win_bias = self.l1_win_bias
        elif layer_idx == 1:
            win_factor = self.l2_win_factor
            win_weight = self.l2_win_weight
            win_bias = self.l2_win_bias

        if modal_win == 2 and modal_len != modal_win:
            modal_step = 1
            modal_len = modal_len - 1
        else:
            modal_step = modal_win

        fusion_node_dict = dict()
        for mo in range(0, modal_len, modal_step):
                basic_node = data_dict[str(mo)]
                for mi in range(mo, mo + modal_win - 1, 1):
                    basic_node = torch.cat((basic_node, data_dict[str(mi+1)]), 1)
                basic_node = basic_node.contiguous().view(batch_size, -1)
                const_vec = Variable((torch.ones(batch_size, 1) * const_bias).type(data_type), requires_grad=False)
                basic_node = torch.cat((const_vec, basic_node), 1)
                basic_node = torch.matmul(basic_node, win_factor[mo])

                poly_node = basic_node
                for _ in range(poly_order):
                    poly_node = poly_node * basic_node

                for i in range(poly_norm):
                    poly_node = (torch.sign(poly_node)) * (torch.sqrt(torch.abs(poly_node)))

                if euclid_norm != 0:
                    n2 = torch.unsqueeze(torch.norm(poly_node.permute(1, 0, 2), dim=2), dim=2)
                    poly_node = torch.div(poly_node.permute(1, 0, 2), n2)
                    poly_node = poly_node.permute(1, 0, 2)

                fusion_node = torch.matmul(win_weight[mo], poly_node.permute(1, 0, 2)).squeeze() + win_bias[mo]
                fusion_node_dict[str(mo)] = fusion_node.view(-1, output_dim)
        return fusion_node_dict

    def __init__(self, poly_order, poly_norm, euclid_norm, input_dims, hidden_dims, text_out, output_dims,
                 init_modal_len, modal_wins, modal_pads, dropouts, rank):
        super(FNL2, self).__init__()
        self.const_bias = 0.5
        self.rank = rank

        self.poly_order = poly_order
        self.poly_norm = poly_norm
        self.euclid_norm = euclid_norm

        audio_in = input_dims[0]
        video_in = input_dims[1]
        text_in = input_dims[2]
        audio_hid = hidden_dims[0]
        video_hid = hidden_dims[1]
        text_hid = hidden_dims[2]
        audio_drop = dropouts[0]
        video_drop = dropouts[1]
        text_drop = dropouts[2]

        self.audio_subnet = SubNet(audio_in, audio_hid, dropout=audio_drop)
        self.video_subnet = SubNet(video_in, video_hid, dropout=video_drop)
        self.text_subnet = TextSubNet(text_in, text_hid, text_out, dropout=text_drop)

        self.modal_lens = []
        self.modal_wins = modal_wins
        self.modal_pads = modal_pads

        self.inter_nodes = []
        self.inter_node_in_dims = dict()
        self.inter_node_out_dims = [output_dims[0], output_dims[1]]

        # construct the first hidden layer
        self.modal_lens.append(init_modal_len + self.modal_pads[0])
        self.inter_nodes.append(self.modal_lens[0] - self.modal_wins[0] + 1)
        if self.modal_wins[0] == 1:
            self.inter_node_in_dims['0'] = [text_out + 1, audio_hid + 1, video_hid + 1]
        elif self.modal_wins[0] == 2:
            self.inter_node_in_dims['0'] = [text_out + audio_hid + 1, audio_hid + video_hid + 1,
                                            video_hid + text_out + 1]
        else:
            self.inter_node_in_dims['0'] = [text_out + audio_hid + video_hid + 1]

        l1_win_factor = []
        l1_win_weight = []
        l1_win_bias = []
        if self.modal_wins[0] == 1 or self.modal_wins[0] == 2:
            for j in range(0, self.inter_nodes[0]):
                    win_factor, win_weight, win_bias = init_window_params(self.rank, self.inter_node_in_dims['0'][j],
                                                                          self.inter_node_out_dims[0])
                    l1_win_factor.append(win_factor)
                    l1_win_weight.append(win_weight)
                    l1_win_bias.append(win_bias)
        else:
            win_factor, win_weight, win_bias = init_window_params(self.rank, self.inter_node_in_dims['0'][0],
                                                                  self.inter_node_out_dims[0])
            l1_win_factor.append(win_factor)
            l1_win_weight.append(win_weight)
            l1_win_bias.append(win_bias)
        self.l1_win_factor = torch.nn.ParameterList(l1_win_factor)
        self.l1_win_weight = torch.nn.ParameterList(l1_win_weight)
        self.l1_win_bias = torch.nn.ParameterList(l1_win_bias)

        # construct the output layer
        self.modal_lens.append(self.modal_lens[0] - self.modal_wins[0] + 1)
        if self.modal_wins[1] == 2:
            self.modal_lens[1] = self.modal_lens[1] + self.modal_pads[1]
        self.inter_nodes.append(self.modal_lens[1] - self.modal_wins[1] + 1)
        self.inter_node_in_dims['1'] = [self.modal_wins[1] * self.inter_node_out_dims[0] + 1]

        l2_win_factor = []
        l2_win_weight = []
        l2_win_bias = []
        win_factor, win_weight, win_bias = init_window_params(self.rank, self.inter_node_in_dims[str(1)][0],
                                                              self.inter_node_out_dims[1])
        l2_win_factor.append(win_factor)
        l2_win_weight.append(win_weight)
        l2_win_bias.append(win_bias)
        self.l2_win_factor = torch.nn.ParameterList(l2_win_factor)
        self.l2_win_weight = torch.nn.ParameterList(l2_win_weight)
        self.l2_win_bias = torch.nn.ParameterList(l2_win_bias)

    def forward(self, audio_x, video_x, text_x):
        audio_hid = self.audio_subnet(audio_x)
        video_hid = self.video_subnet(video_x)
        text_hid = self.text_subnet(text_x)

        data_dict = dict()
        if audio_hid.is_cuda:
            data_type = torch.cuda.FloatTensor
        else:
            data_type = torch.FloatTensor

        if self.modal_wins[0] == 2:
            # copy and pad the first node
            data_dict['0'] = text_hid
            data_dict['1'] = audio_hid
            data_dict['2'] = video_hid
            data_dict['3'] = text_hid
        else:
            data_dict['0'] = text_hid
            data_dict['1'] = audio_hid
            data_dict['2'] = video_hid

        # fuse to produce first hidden layer
        inter_node_dict = self.ptp_layer(data_dict, data_type, 0)

        # fuse to produce output layer
        if self.modal_wins[1] == 2:
            inter_node_dict['3'] = inter_node_dict['0']

        final_node_dict = self.ptp_layer(inter_node_dict, data_type, 1)
        final_node_dict = final_node_dict['0'].squeeze()
        return final_node_dict
