import os
import argparse
import random
import numpy as np
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils import total, load_iemocap
from sklearn.metrics import accuracy_score, f1_score
from model_iecomcap_l2 import FNL2


def display(f1, acc):
    print('Test f1 score: {}'.format(f1))
    print('Test accuracy: {}'.format(acc))


def main(opts):
    data_type_float = torch.FloatTensor
    data_type_long = torch.LongTensor

    # load option configs
    emotion = opts['emotion']

    poly_order = opts['poly_order']
    poly_norm = opts['poly_norm']
    euclid_norm = opts['euclid_norm']

    init_modal_len = opts['init_modal_len']
    modal_wins = opts['modal_wins']
    modal_pads = opts['modal_pads']
    output_dim = opts['output_dim']

    epochs = opts['epochs']
    patience = opts['patience']

    signature = opts['signature']
    run_id = opts['run_id']

    data_path = opts['data_path']
    model_path = opts['model_path']
    output_path = opts['output_path']

    # set paths for storing models and outputs
    model_path = os.path.join(model_path, "model_{}_{}_{}.pt".format(signature, emotion, run_id))
    output_path = os.path.join(output_path, "result_{}_{}_{}.csv".format(signature, emotion, run_id))

    if not os.path.isfile(output_path):
        with open(output_path, 'w+') as out:
            writer = csv.writer(out)
            writer.writerow(['audio_hidden', 'video_hidden', 'text_hidden',
                             'audio_dropout', 'video_dropout', 'text_dropout',
                             'factor_learning_rate', 'learning_rate', 'rank', 'batch_size', 'weight_decay',
                             'best_validation_loss', 'test_loss', 'test_f1', 'test_acc', 'hid1_outdim'])

    # load data sets
    train_set, valid_set, test_set, input_dims = load_iemocap(data_path, emotion)

    params = dict()
    params['audio_hidden'] = [8, 16, 32]
    params['video_hidden'] = [4, 8, 16]
    params['text_hidden'] = [64, 128, 256]
    params['audio_dropout'] = [0, 0.1, 0.15, 0.2, 0.3, 0.5]
    params['video_dropout'] = [0, 0.1, 0.15, 0.2, 0.3, 0.5]
    params['text_dropout'] = [0, 0.1, 0.15, 0.2, 0.3, 0.5]
    params['factor_learning_rate'] = [0.0003, 0.0005, 0.001, 0.003]
    params['learning_rate'] = [0.0003, 0.0005, 0.001, 0.003]
    params['rank'] = [1, 4, 8, 16]
    params['batch_size'] = [8, 16, 32, 64, 128]
    params['weight_decay'] = [0, 0.001, 0.002, 0.01]
    params['hid1_out_dim'] = [30, 40, 50, 60, 70]

    total_settings = total(params)
    seen_settings = set()
    print('There are {} different hyper parameter settings in total.'.format(total_settings))

    for i in range(total_settings):
        audio_hidden = random.choice(params['audio_hidden'])
        video_hidden = random.choice(params['video_hidden'])
        text_hidden = random.choice(params['text_hidden'])
        text_out = text_hidden // 2
        audio_dropout = random.choice(params['audio_dropout'])
        video_dropout = random.choice(params['video_dropout'])
        text_dropout = random.choice(params['text_dropout'])
        factor_lr = random.choice(params['factor_learning_rate'])
        lr = random.choice(params['learning_rate'])
        rank = random.choice(params['rank'])
        batch_size = random.choice(params['batch_size'])
        weight_decay = random.choice(params['weight_decay'])
        hid1_out_dim = random.choice(params['hid1_out_dim'])

        # reject the setting if it has been tried
        current_setting = (audio_hidden, video_hidden, text_hidden, audio_dropout, video_dropout, text_dropout,
                           factor_lr, lr, rank, batch_size, weight_decay, hid1_out_dim)
        if current_setting in seen_settings:
            continue
        else:
            seen_settings.add(current_setting)

        model = FNL2(poly_order, poly_norm, euclid_norm,
                     input_dims, (audio_hidden, video_hidden, text_hidden), text_out, (hid1_out_dim, output_dim),
                     init_modal_len, modal_wins, modal_pads,
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
            print('Model in cuda')
            model = model.cuda()
            data_type_float = torch.cuda.FloatTensor
            data_type_long = torch.cuda.LongTensor
        print('Model initialized')

        # loss and optimizer
        criterion = nn.CrossEntropyLoss(size_average=False)
        optimizer = optim.Adam([{'params': win_factors, 'lr': factor_lr}, {'params': other_params, 'lr': lr}],
                               weight_decay=weight_decay)

        # setup training
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
                x_a = Variable(x[0].float().type(data_type_float), requires_grad=False)
                x_v = Variable(x[1].float().type(data_type_float), requires_grad=False)
                x_t = Variable(x[2].float().type(data_type_float), requires_grad=False)
                y = Variable(batch[-1].view(-1, output_dim).float().type(data_type_long), requires_grad=False)
                try:
                    output = model(x_a, x_v, x_t)
                except ValueError as e:
                    print(x_a.data.shape)
                    print(x_v.data.shape)
                    print(x_t.data.shape)
                    raise e
                loss = criterion(output, torch.max(y, 1)[1])
                loss.backward()
                avg_loss = loss.item()
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
                x_a = Variable(x[0].float().type(data_type_float), requires_grad=False)
                x_v = Variable(x[1].float().type(data_type_float), requires_grad=False)
                x_t = Variable(x[2].float().type(data_type_float), requires_grad=False)
                y = Variable(batch[-1].view(-1, output_dim).float().type(data_type_long), requires_grad=False)
                output = model(x_a, x_v, x_t)
                valid_loss = criterion(output, torch.max(y, 1)[1])
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
            test_loss = None
            best_model = torch.load(model_path)
            best_model.eval()
            for batch in test_iter:
                x = batch[:-1]
                x_a = Variable(x[0].float().type(data_type_float), requires_grad=False)
                x_v = Variable(x[1].float().type(data_type_float), requires_grad=False)
                x_t = Variable(x[2].float().type(data_type_float), requires_grad=False)
                y = Variable(batch[-1].view(-1, output_dim).float().type(data_type_long), requires_grad=False)
                output_test = best_model(x_a, x_v, x_t)
                loss_test = criterion(output_test, torch.max(y, 1)[1])
                test_loss = loss_test.item()
            output_test = output_test.cpu().data.numpy().reshape(-1, output_dim)
            y = y.cpu().data.numpy().reshape(-1, output_dim)
            test_loss = test_loss / len(test_set)

            all_true_label = np.argmax(y, axis=1)
            all_predicted_label = np.argmax(output_test, axis=1)
            f1 = f1_score(all_true_label, all_predicted_label, average='weighted')
            acc_score = accuracy_score(all_true_label, all_predicted_label)
            display(f1, acc_score)

            with open(output_path, 'a+') as out:
                writer = csv.writer(out)
                writer.writerow([audio_hidden, video_hidden, text_hidden,
                                 audio_dropout, video_dropout, text_dropout,
                                 factor_lr, lr, rank, batch_size, weight_decay,
                                 min_valid_loss, test_loss, f1, acc_score, hid1_out_dim])


if __name__ == "__main__":
    opts_config = argparse.ArgumentParser()
    opts_config.add_argument('--emotion', dest='emotion', type=str, default='angry')

    opts_config.add_argument('--poly_order', dest='poly_order', type=list, default=[2, 2],
                             help='the polynomial order of the each layer')
    opts_config.add_argument('--poly_norm', dest='poly_norm', type=int, default=1,
                             help='the polynomial normalization')
    opts_config.add_argument('--euclid_norm', dest='euclid_norm', type=int, default=0,
                             help='the l2 normalization')

    opts_config.add_argument('--init_modal_len', dest='init_modal_len', type=int, default=3,
                             help='the number of modalities of the input layer')
    opts_config.add_argument('--modal_wins', dest='modal_wins', type=list, default=[2, 3],
                             help='the modality window sizes from the input layer to the last hidden layer')
    opts_config.add_argument('--modal_pads', dest='modal_pads', type=list, default=[1, 0],
                             help='the modality pad sizes from the input layer to the last hidden layer')

    opts_config.add_argument('--epochs', dest='epochs', type=int, default=10)
    opts_config.add_argument('--patience', dest='patience', type=int, default=20)
    opts_config.add_argument('--output_dim', dest='output_dim', type=int, default=2)

    opts_config.add_argument('--run_id', dest='run_id', type=int, default=1)
    opts_config.add_argument('--signature', dest='signature', type=str, default='iemocap')

    opts_config.add_argument('--data_path', dest='data_path', type=str, default='./data/')
    opts_config.add_argument('--model_path', dest='model_path', type=str, default='./models/')
    opts_config.add_argument('--output_path', dest='output_path', type=str, default='./results/')

    opts_config = vars(opts_config.parse_args())
    main(opts_config)
