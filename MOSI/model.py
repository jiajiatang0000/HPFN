import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.init import xavier_normal_


# initialize parameters for scanning window
def init_window_params(rank, hidden_dim, out_dim):
    win_factor = torch.nn.Parameter(torch.Tensor(rank, hidden_dim, out_dim))
    win_weight = torch.nn.Parameter(torch.Tensor(1, rank))
    win_bias = torch.nn.Parameter(torch.Tensor(1, out_dim))
    win_factor = xavier_normal_(win_factor)
    win_weight = xavier_normal_(win_weight)
    win_bias.data.fill_(0)
    return win_factor, win_weight, win_bias


class SubNet(nn.Module):
    def __init__(self, in_size, hidden_size, num_layers=1, dropout=0.2, bidirectional=False):
        super(SubNet, self).__init__()
        self.rnn = nn.LSTM(in_size, hidden_size,
                           num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        all_states, _ = self.rnn(x)
        state_shape = all_states.shape
        all_states = torch.reshape(all_states, [-1, state_shape[-1]])
        all_h = self.dropout(all_states)
        all_y = torch.reshape(all_h, [state_shape[0], state_shape[1], -1])
        return all_y


class FNL2NO(nn.Module):
    """
    HPFN model two-layer non-overlap
    """
    # ptp operations in the specific layer
    def ptp_layer(self, data_dict, data_type, layer_idx):
        const_bias = self.const_bias
        batch_size = data_dict['0'].shape[0]

        time_len = data_dict['0'].shape[1]
        modal_len = self.modal_lens[layer_idx]

        time_win = self.time_wins[layer_idx]
        modal_win = self.modal_wins[layer_idx]

        output_dim = self.inter_node_out_dims[layer_idx]
        poly_order = self.poly_order[layer_idx]
        poly_norm = self.poly_norm
        euclid_norm = self.euclid_norm

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

        # for the case like when all_modal = 4 & modal_step = 2
        if modal_win == 2 and modal_len != modal_win:
            modal_step = 1
            modal_len = modal_len - 1
        else:
            modal_step = modal_win

        # generate the fusion nodes of next layer
        fusion_node_dict = dict()
        ni = 0
        # mo indexing the window along the modality domain
        for mo in range(0, modal_len, modal_step):
            fusion_nodes = []
            # t indexing the window along the time domain
            for t in range(0, time_len, time_win):
                # concatenate all the nodes within a window
                basic_node = data_dict[str(mo)][:, t: t + time_win, :]
                # mo indexing the node within a window
                for mi in range(mo, mo + modal_win - 1, 1):
                    basic_node = torch.cat((basic_node, data_dict[str(mi + 1)][:, t: t + time_win, :]), 2)
                basic_node = basic_node.contiguous().view(batch_size, -1)
                # concatenate an extra constant vector
                const_vec = Variable((torch.ones(batch_size, 1) * const_bias).type(data_type), requires_grad=False)
                basic_node = torch.cat((const_vec, basic_node), 1)

                # weight the basic node
                basic_node = torch.matmul(basic_node, win_factor[ni])

                # power the basic node
                poly_node = basic_node
                for _ in range(poly_order):
                    poly_node = poly_node * basic_node

                # polynomial normalization
                for i in range(poly_norm):
                    poly_node = (torch.sign(poly_node)) * (torch.sqrt(torch.abs(poly_node)))

                # euclidean normalization
                if euclid_norm != 0:
                    n2 = torch.unsqueeze(torch.norm(poly_node.permute(1, 0, 2), dim=2), dim=2)
                    poly_node = torch.div(poly_node.permute(1, 0, 2), n2)
                    poly_node = poly_node.permute(1, 0, 2)

                # non-linear transformation
                fusion_node = torch.matmul(win_weight[ni], poly_node.permute(1, 0, 2)).squeeze() + win_bias[ni]
                fusion_node = fusion_node.view(-1, output_dim)

                fusion_nodes.append(fusion_node)
                ni = ni + 1
            fusion_node_dict[str(mo)] = torch.stack(fusion_nodes, dim=2).permute(0, 2, 1)
        return fusion_node_dict

    def __init__(self, poly_order, poly_norm, euclid_norm, input_dims, hidden_dims, output_dims,
                 init_time_len, init_time_win, time_wins, time_pads, init_modal_len, modal_wins, modal_pads,
                 dropouts, rank):
        """
        Args:
            input_dims - a length-3 tuple, contains (audio_dim, video_dim, text_dim)
            hidden_dims - a length-3 tuple, hidden dims of the sub-networks
            output_dims - a length-2 tuple, specifying the sizes of outputs
            dropouts - a length-4 tuple, contains (audio_dropout, video_dropout, text_dropout, post_fusion_dropout)
            rank - int, specifying the size of rank
        Output:
            a scalar value between -3 and 3
        """
        super(FNL2NO, self).__init__()
        self.const_bias = 0.5
        self.rank = rank

        self.poly_order = poly_order
        self.poly_norm = poly_norm
        self.euclid_norm = euclid_norm

        # the time dimensions of all layers
        self.time_lens = []
        # the time window sizes from the input layer to the last hidden layer
        self.time_wins = []
        if len(time_wins) == 0:
            self.time_wins.append(init_time_win)
            self.time_wins.append(init_time_len // init_time_win)
        else:
            self.time_wins = time_wins

        # the time pad sizes from the input layer to the last hidden layer
        self.time_pads = time_pads

        # the number of modalities of all layers
        self.modal_lens = []
        # the modality window sizes from the input layer to the last hidden layer
        self.modal_wins = modal_wins
        # the modality pad sizes from the input layer to the last hidden layer
        self.modal_pads = modal_pads

        # the number of intermediate nodes from the first hidden layer to the output layer
        self.inter_nodes_rows = []
        self.inter_nodes_cols = []
        self.inter_nodes = []
        # the dimensions of intermediate nodes from the first hidden layer to the output layer
        self.inter_node_in_dims = dict()
        # the output dimension of the node based on the ptp operator
        self.inter_node_out_dims = [output_dims[0], output_dims[1]]

        # process input features using rnn
        self.audio_in = input_dims[0]
        self.video_in = input_dims[1]
        self.text_in = input_dims[2]
        audio_hid = hidden_dims[0]
        video_hid = hidden_dims[1]
        text_hid = hidden_dims[2]
        self.audio_drop = dropouts[0]
        self.video_drop = dropouts[1]
        self.text_drop = dropouts[2]
        self.audio_subnet = SubNet(self.audio_in, audio_hid, dropout=self.audio_drop)
        self.video_subnet = SubNet(self.video_in, video_hid, dropout=self.video_drop)
        self.text_subnet = SubNet(self.text_in, text_hid, dropout=self.text_drop)

        # construct the first hidden layer
        self.time_lens.append(init_time_len + self.time_pads[0])
        self.modal_lens.append(init_modal_len + self.modal_pads[0])

        self.inter_nodes_rows.append(self.time_lens[0] // self.time_wins[0])
        self.inter_nodes_cols.append(self.modal_lens[0] - self.modal_wins[0] + 1)
        self.inter_nodes.append(self.inter_nodes_rows[0] * self.inter_nodes_cols[0])

        if self.modal_wins[0] == 1:
            self.inter_node_in_dims['0'] = [self.time_wins[0] * text_hid + 1,
                                            self.time_wins[0] * audio_hid + 1,
                                            self.time_wins[0] * video_hid + 1]
        elif self.modal_wins[0] == 2:
            self.inter_node_in_dims['0'] = [self.time_wins[0] * (text_hid + audio_hid) + 1,
                                            self.time_wins[0] * (audio_hid + video_hid) + 1,
                                            self.time_wins[0] * (video_hid + text_hid) + 1]
        else:
            self.inter_node_in_dims['0'] = [self.time_wins[0] * (text_hid + audio_hid + video_hid) + 1]

        l1_win_factor = []
        l1_win_weight = []
        l1_win_bias = []
        if self.modal_wins[0] == 1 or self.modal_wins[0] == 2:
            for j in range(0, self.inter_nodes_cols[0]):
                for _ in range(0, self.time_lens[0], self.time_wins[0]):
                    win_factor, win_weight, win_bias = init_window_params(self.rank, self.inter_node_in_dims['0'][j],
                                                                          self.inter_node_out_dims[0])
                    l1_win_factor.append(win_factor)
                    l1_win_weight.append(win_weight)
                    l1_win_bias.append(win_bias)
        else:
            for _ in range(0, self.time_lens[0], self.time_wins[0]):
                win_factor, win_weight, win_bias = init_window_params(self.rank, self.inter_node_in_dims['0'][0],
                                                                      self.inter_node_out_dims[0])
                l1_win_factor.append(win_factor)
                l1_win_weight.append(win_weight)
                l1_win_bias.append(win_bias)
        self.l1_win_factor = torch.nn.ParameterList(l1_win_factor)
        self.l1_win_weight = torch.nn.ParameterList(l1_win_weight)
        self.l1_win_bias = torch.nn.ParameterList(l1_win_bias)

        # construct the output layer
        self.time_lens.append(self.time_lens[0] // self.time_wins[0])
        self.modal_lens.append(self.modal_lens[0] - self.modal_wins[0] + 1)
        if self.modal_wins[1] == 2:
            self.modal_lens[1] = self.modal_lens[1] + self.modal_pads[1]

        self.inter_nodes_rows.append(self.time_lens[1] // self.time_wins[1])
        self.inter_nodes_cols.append(self.modal_lens[1] - self.modal_wins[1] + 1)
        self.inter_nodes.append(self.inter_nodes_rows[1] * self.inter_nodes_cols[1])
        self.inter_node_in_dims['1'] = [(self.time_wins[1] * self.modal_wins[1]) * self.inter_node_out_dims[0] + 1]

        l2_win_factor = []
        l2_win_weight = []
        l2_win_bias = []
        for _ in range(0, self.time_lens[1], self.time_wins[1]):
            win_factor, win_weight, win_bias = init_window_params(self.rank, self.inter_node_in_dims[str(1)][0],
                                                                  self.inter_node_out_dims[1])
            l2_win_factor.append(win_factor)
            l2_win_weight.append(win_weight)
            l2_win_bias.append(win_bias)
        self.l2_win_factor = torch.nn.ParameterList(l2_win_factor)
        self.l2_win_weight = torch.nn.ParameterList(l2_win_weight)
        self.l2_win_bias = torch.nn.ParameterList(l2_win_bias)

    def forward(self, audio_x, video_x, text_x):
        """
        Args:
            audio_x: tensor of shape (batch_size, audio_in)
            video_x: tensor of shape (batch_size, video_in)
            text_x: tensor of shape (batch_size, sequence_len, text_in)
        """
        # the hidden states of the sub_network
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
        inter_node_dict = self.ptp_layer(data_dict, data_type, layer_idx=0)

        # fuse to produce output layer
        if self.modal_wins[1] == 2:
            inter_node_dict['3'] = inter_node_dict['0']
        final_node_dict = self.ptp_layer(inter_node_dict, data_type, layer_idx=1)
        final_node_dict = final_node_dict['0'].squeeze(2)
        return final_node_dict
