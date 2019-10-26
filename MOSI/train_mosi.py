import os
import argparse
import random
import numpy as np
import h5py
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from utils import total
from sklearn.metrics import accuracy_score, f1_score
from model import FNL2NO


def display(mae, corr, multi_acc, bi_acc, f1):
    print('Test mae: {}'.format(mae))
    print('Test correlation: {}'.format(corr))
    print('Test multi-class accuracy: {}'.format(multi_acc))
    print('Test binary accuracy: {}'.format(bi_acc))
    print('Test f1 score: {}'.format(f1))
    print('\n')


def load_saved_data(input_dims):
    class LoadDataSet(Dataset):
        def __init__(self, audio, visual, text, labels):
            self.audio = audio
            self.visual = visual
            self.text = text
            self.labels = labels

        def __getitem__(self, idx):
            return [self.audio[idx, :, :], self.visual[idx, :, :], self.text[idx, :, :], self.labels[idx]]

        def __len__(self):
            return self.audio.shape[0]

    h5f = h5py.File('./data/X_train.h5', 'r')
    x_train = h5f['data'][:]
    h5f.close()
    h5f = h5py.File('./data/y_train.h5', 'r')
    y_train = h5f['data'][:]
    h5f.close()
    h5f = h5py.File('./data/X_valid.h5', 'r')
    x_valid = h5f['data'][:]
    h5f.close()
    h5f = h5py.File('./data/y_valid.h5', 'r')
    y_valid = h5f['data'][:]
    h5f.close()
    h5f = h5py.File('./data/X_test.h5', 'r')
    x_test = h5f['data'][:]
    h5f.close()
    h5f = h5py.File('./data/y_test.h5', 'r')
    y_test = h5f['data'][:]
    h5f.close()

    ad = input_dims[0]
    td = input_dims[2]
    train_set = LoadDataSet(x_train[:, :, td:td + ad], x_train[:, :, td + ad:], x_train[:, :, :td], y_train)
    valid_set = LoadDataSet(x_valid[:, :, td:td + ad], x_valid[:, :, td + ad:], x_valid[:, :, :td], y_valid)
    test_set = LoadDataSet(x_test[:, :, td:td + ad], x_test[:, :, td + ad:], x_test[:, :, :td], y_test)
    return train_set, y_train, valid_set, y_valid, test_set, y_test


def main(opts):
    data_type = torch.FloatTensor

    # load option configs
    poly_order = opts['poly_order']
    poly_norm = opts['poly_norm']
    euclid_norm = opts['euclid_norm']

    audio_dim = opts['audio_dim']
    video_dim = opts['video_dim']
    text_dim = opts['text_dim']

    input_dims = (audio_dim, video_dim, text_dim)
    output_dim = opts['output_dim']

    init_modal_len = opts['init_modal_len']
    modal_wins = opts['modal_wins']
    modal_pads = opts['modal_pads']

    init_time_len = opts['init_time_len']
    time_wins = opts['time_wins']
    time_pads = opts['time_pads']

    epochs = opts['epochs']
    patience = opts['patience']

    signature = opts['signature']
    run_id = opts['run_id']
    model_path = opts['model_path']
    output_path = opts['output_path']

    # set paths for storing models and outputs
    model_path = os.path.join(model_path, 'model_{}_{}.pt'.format(signature, run_id))
    output_path = os.path.join(output_path, 'result_{}_{}.csv'.format(signature, run_id))

    # load data sets
    train_set, label_train, valid_set, label_valid, test_set, label_test = load_saved_data(input_dims)

    # set parameters
    params = dict()
    params['audio_hidden'] = [4, 8, 16]
    params['video_hidden'] = [4, 8, 16]
    params['text_hidden'] = [64, 128, 256]
    params['audio_dropout'] = [0, 0.1, 0.15, 0.2, 0.3, 0.5]
    params['video_dropout'] = [0, 0.1, 0.15, 0.2, 0.3, 0.5]
    params['text_dropout'] = [0, 0.1, 0.15, 0.2, 0.3, 0.5]
    params['factor_learning_rate'] = [0.0003, 0.0005, 0.001, 0.003]
    params['learning_rate'] = [0.0003, 0.0005, 0.001, 0.003]
    params['rank'] = [1, 4, 8, 16]
    params['batch_size'] = [4, 8, 16, 32, 64, 128]
    params['weight_decay'] = [0, 0.001, 0.002, 0.01]
    # the window size in input layer
    params['init_time_win'] = [1, 2, 4, 5, 10, 20]
    # the output dimensions of the ptp block in first hidden layer
    params['hid1_out_dim'] = [30, 40, 50, 60, 70]

    total_settings = total(params)
    seen_settings = set()
    print('There are {} different hyper parameter settings in total.'.format(total_settings))

    with open(output_path, 'w+') as out:
        writer = csv.writer(out)
        writer.writerow(['audio_hidden', 'video_hidden', 'text_hidden',
                         'audio_dropout', 'video_dropout', 'text_dropout',
                         'factor_learning_rate', 'learning_rate', 'rank', 'batch_size', 'weight_decay',
                         'best_valid_mae', 'test_mae', 'test_corr',
                         'test_multi_acc', 'test_binary_acc', 'test_f1',
                         'init_time_win', 'hid1_out_dim'])

    for i in range(total_settings):
        audio_hidden = random.choice(params['audio_hidden'])
        video_hidden = random.choice(params['video_hidden'])
        text_hidden = random.choice(params['text_hidden'])

        audio_dropout = random.choice(params['audio_dropout'])
        video_dropout = random.choice(params['video_dropout'])
        text_dropout = random.choice(params['text_dropout'])

        factor_lr = random.choice(params['factor_learning_rate'])
        lr = random.choice(params['learning_rate'])
        rank = random.choice(params['rank'])
        batch_size = random.choice(params['batch_size'])
        weight_decay = random.choice(params['weight_decay'])

        init_time_win = random.choice(params['init_time_win'])
        hid1_out_dim = random.choice(params['hid1_out_dim'])

        # reject the setting if it has been tried
        current_setting = (audio_hidden, video_hidden, text_hidden, audio_dropout, video_dropout, text_dropout,
                           factor_lr, lr, rank, batch_size, weight_decay, init_time_win, hid1_out_dim)
        if current_setting in seen_settings:
            continue
        else:
            seen_settings.add(current_setting)

        # initialize the model
        model = FNL2NO(poly_order, poly_norm, euclid_norm,
                       input_dims, (audio_hidden, video_hidden, text_hidden), (hid1_out_dim, output_dim),
                       init_time_len, init_time_win, time_wins, time_pads, init_modal_len, modal_wins, modal_pads,
                       (audio_dropout, video_dropout, text_dropout), rank)

        win_factor_list = []
        for n in range(model.inter_nodes[0]):
            win_factor_list.append('l1_win_factor.' + str(n))
        win_factor_list.append('l2_win_factor.0')

        # split the parameters of the model into two parts
        win_factors = []
        other_params = []
        for name, param in model.named_parameters():
            if name in win_factor_list:
                win_factors.append(param)
            else:
                other_params.append(param)
        if torch.cuda.is_available():
            model = model.cuda()
            data_type = torch.cuda.FloatTensor
        print('Model initialized')

        # loss and optimizer
        criterion = nn.L1Loss(size_average=False)
        optimizer = optim.Adam([{'params': win_factors, 'lr': factor_lr}, {'params': other_params, 'lr': lr}],
                               weight_decay=weight_decay)

        complete = True
        min_valid_loss = float('Inf')
        train_iter = DataLoader(train_set, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)
        valid_iter = DataLoader(valid_set, batch_size=len(valid_set), num_workers=4, shuffle=True)
        test_iter = DataLoader(test_set, batch_size=len(test_set), num_workers=4, shuffle=True)
        curr_patience = patience

        for epoch in range(epochs):
            model.train()
            model.zero_grad()
            avg_train_loss = 0

            for batch in train_iter:
                model.zero_grad()
                x = batch[:-1]
                x_a = Variable(x[0].float().type(data_type), requires_grad=False)
                x_v = Variable(x[1].float().type(data_type), requires_grad=False)
                x_t = Variable(x[2].float().type(data_type), requires_grad=False)
                y = Variable(batch[-1].view(-1, output_dim).float().type(data_type), requires_grad=False)
                output = model(x_a, x_v, x_t)
                loss = criterion(output, y)
                loss.backward()
                avg_loss = loss.data.item()
                avg_train_loss += avg_loss / len(train_set)
                optimizer.step()
            print('Epoch {}'.format(epoch))
            print('Training loss: {}'.format(avg_train_loss))

            # terminate the training process if run into nan
            if np.isnan(avg_train_loss):
                print('Training got into NaN values...\n\n')
                complete = False
                break

            model.eval()
            avg_valid_loss = 0
            for batch in valid_iter:
                x = batch[:-1]
                x_a = Variable(x[0].float().type(data_type), requires_grad=False).squeeze()
                x_v = Variable(x[1].float().type(data_type), requires_grad=False).squeeze()
                x_t = Variable(x[2].float().type(data_type), requires_grad=False)
                y = Variable(batch[-1].view(-1, output_dim).float().type(data_type), requires_grad=False)
                output = model(x_a, x_v, x_t)
                valid_loss = criterion(output, y)
                avg_valid_loss = valid_loss.item()

            if np.isnan(avg_valid_loss):
                print('Validation got into NaN values...\n\n')
                complete = False
                break

            avg_valid_loss = avg_valid_loss / len(valid_set)
            print('Validation loss: {}'.format(avg_valid_loss))
            if avg_valid_loss < min_valid_loss:
                curr_patience = patience
                min_valid_loss = avg_valid_loss
                torch.save(model, model_path)
                print('Found new best model, saving to disk...')
            else:
                curr_patience -= 1

            if curr_patience <= 0:
                break
            print('\n\n')

        if complete:
            output_test = None
            y = None
            best_model = torch.load(model_path)
            best_model.eval()
            for batch in test_iter:
                x = batch[:-1]
                x_a = Variable(x[0].float().type(data_type), requires_grad=False).squeeze()
                x_v = Variable(x[1].float().type(data_type), requires_grad=False).squeeze()
                x_t = Variable(x[2].float().type(data_type), requires_grad=False)
                y = Variable(batch[-1].view(-1, output_dim).float().type(data_type), requires_grad=False)
                output_test = best_model(x_a, x_v, x_t)

            output_test = output_test.cpu().data.numpy().reshape(-1, output_dim)
            y = y.cpu().data.numpy().reshape(-1, output_dim)
            output_test = output_test.reshape((len(output_test),))
            y = y.reshape((len(y),))

            mae = np.mean(np.absolute(output_test - y))
            corr = np.corrcoef(output_test, y)[0][1]
            multi_acc = round(sum(np.round(output_test) == np.round(y)) / float(len(y)), 5)
            true_label = (y >= 0)
            predicted_label = (output_test >= 0)
            bi_acc = accuracy_score(true_label, predicted_label)
            f1 = f1_score(true_label, predicted_label, average='weighted')
            display(mae, corr, multi_acc, bi_acc, f1)

            with open(output_path, 'a+') as out:
                writer = csv.writer(out)
                writer.writerow([audio_hidden, video_hidden, text_hidden,
                                 audio_dropout, video_dropout, text_dropout,
                                 factor_lr, lr, rank, batch_size, weight_decay,
                                 min_valid_loss, mae, corr, multi_acc, bi_acc, f1])


if __name__ == "__main__":
    opts_config = argparse.ArgumentParser()
    opts_config.add_argument('--poly_order', dest='poly_order', type=list, default=[2, 2],
                             help='the polynomial order of the each layer')
    opts_config.add_argument('--poly_norm', dest='poly_norm', type=int, default=1,
                             help='the polynomial normalization')
    opts_config.add_argument('--euclid_norm', dest='euclid_norm', type=int, default=0,
                             help='the l2 normalization')

    opts_config.add_argument('--audio_dim', dest='audio_dim', type=int, default=5,
                             help='the input audio dimension')
    opts_config.add_argument('--video_dim', dest='video_dim', type=int, default=20,
                             help='the input video dimension')
    opts_config.add_argument('--text_dim', dest='text_dim', type=int, default=300,
                             help='the input text dimension')

    opts_config.add_argument('--init_time_len', dest='init_time_len', type=int, default=20,
                             help='the number of time steps of the input layer')
    opts_config.add_argument('--time_wins', dest='time_wins', type=list, default=[4, 5],
                             help='the time window sizes from the input layer to the last hidden layer')
    opts_config.add_argument('--time_pads', dest='time_pads', type=list, default=[0, 0],
                             help='the time pad sizes from the input layer to the last hidden layer')
    opts_config.add_argument('--init_modal_len', dest='init_modal_len', type=int, default=3,
                             help='the number of modalities of the input layer')
    opts_config.add_argument('--modal_wins', dest='modal_wins', type=list, default=[2, 3],
                             help='the modality window sizes from the input layer to the last hidden layer')
    opts_config.add_argument('--modal_pads', dest='modal_pads', type=list, default=[1, 0],
                             help='the modality pad sizes from the input layer to the last hidden layer')

    opts_config.add_argument('--epochs', dest='epochs', type=int, default=10)
    opts_config.add_argument('--patience', dest='patience', type=int, default=20)
    opts_config.add_argument('--output_dim', dest='output_dim', type=int, default=1)

    opts_config.add_argument('--run_id', dest='run_id', type=int, default=1)
    opts_config.add_argument('--signature', dest='signature', type=str, default='mosi')

    opts_config.add_argument('--data_path', dest='data_path', type=str, default='./data/')
    opts_config.add_argument('--model_path', dest='model_path', type=str, default='./models/')
    opts_config.add_argument('--output_path', dest='output_path', type=str, default='./results/')

    opts_config = vars(opts_config.parse_args())
    main(opts_config)
