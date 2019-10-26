from torch.utils.data import Dataset
import pickle as pickle
import numpy as np

AUDIO = 'covarep'
VISUAL = 'facet'
TEXT = 'glove'
LABEL = 'label'

def total(params):
    '''
    count the total number of hyperparameter settings
    '''
    settings = 1
    for k, v in params.items():
        settings *= len(v)
    return settings

def load_pom(data_path):
    # parse the input args
    class POM(Dataset):
        '''
        PyTorch Dataset for POM, don't need to change this
        '''
        def __init__(self, audio, visual, text, labels):
            self.audio = audio
            self.visual = visual
            self.text = text
            self.labels = labels

        def __getitem__(self, idx):
            return [self.audio[idx, :], self.visual[idx, :], self.text[idx, :, :], self.labels[idx]]

        def __len__(self):
            return self.audio.shape[0]

    pom_data = pickle.load(open(data_path + "pom.pkl", 'rb'), encoding='latin1')
    pom_train, pom_valid, pom_test = pom_data['train'], pom_data['valid'], pom_data['test']

    train_audio, train_visual, train_text, train_labels \
        = pom_train[AUDIO], pom_train[VISUAL], pom_train[TEXT], pom_train[LABEL]
    valid_audio, valid_visual, valid_text, valid_labels \
        = pom_valid[AUDIO], pom_valid[VISUAL], pom_valid[TEXT], pom_valid[LABEL]
    test_audio, test_visual, test_text, test_labels \
        = pom_test[AUDIO], pom_test[VISUAL], pom_test[TEXT], pom_test[LABEL]

    # code that instantiates the Dataset objects
    train_set = POM(train_audio, train_visual, train_text, train_labels)
    valid_set = POM(valid_audio, valid_visual, valid_text, valid_labels)
    test_set = POM(test_audio, test_visual, test_text, test_labels)


    audio_dim = train_set[0][0].shape[0]
    print("Audio feature dimension is: {}".format(audio_dim))
    visual_dim = train_set[0][1].shape[0]
    print("Visual feature dimension is: {}".format(visual_dim))
    text_dim = train_set[0][2].shape[1]
    print("Text feature dimension is: {}".format(text_dim))
    input_dims = (audio_dim, visual_dim, text_dim)

    # remove possible NaN values
    train_set.visual[train_set.visual != train_set.visual] = 0
    valid_set.visual[valid_set.visual != valid_set.visual] = 0
    test_set.visual[test_set.visual != test_set.visual] = 0

    train_set.audio[train_set.audio != train_set.audio] = 0
    valid_set.audio[valid_set.audio != valid_set.audio] = 0
    test_set.audio[test_set.audio != test_set.audio] = 0

    return train_set, valid_set, test_set, input_dims

def load_iemocap(data_path, emotion):
    # parse the input args
    class IEMOCAP(Dataset):
        '''
        PyTorch Dataset for IEMOCAP, don't need to change this
        '''
        def __init__(self, audio, visual, text, labels):
            self.audio = audio
            self.visual = visual
            self.text = text
            self.labels = labels

        def __getitem__(self, idx):
            return [self.audio[idx, :], self.visual[idx, :], self.text[idx, :, :], self.labels[idx]]

        def __len__(self):
            return self.audio.shape[0]


    iemocap_data = pickle.load(open(data_path + "iemocap.pkl", 'rb'), encoding='latin1')
    iemocap_train, iemocap_valid, iemocap_test = iemocap_data[emotion]['train'], iemocap_data[emotion]['valid'], iemocap_data[emotion]['test']

    train_audio, train_visual, train_text, train_labels \
        = iemocap_train[AUDIO], iemocap_train[VISUAL], iemocap_train[TEXT], iemocap_train[LABEL]
    valid_audio, valid_visual, valid_text, valid_labels \
        = iemocap_valid[AUDIO], iemocap_valid[VISUAL], iemocap_valid[TEXT], iemocap_valid[LABEL]
    test_audio, test_visual, test_text, test_labels \
        = iemocap_test[AUDIO], iemocap_test[VISUAL], iemocap_test[TEXT], iemocap_test[LABEL]

    # code that instantiates the Dataset objects
    train_set = IEMOCAP(train_audio, train_visual, train_text, train_labels)
    valid_set = IEMOCAP(valid_audio, valid_visual, valid_text, valid_labels)
    test_set = IEMOCAP(test_audio, test_visual, test_text, test_labels)


    audio_dim = train_set[0][0].shape[0]
    print("Audio feature dimension is: {}".format(audio_dim))
    visual_dim = train_set[0][1].shape[0]
    print("Visual feature dimension is: {}".format(visual_dim))
    text_dim = train_set[0][2].shape[1]
    print("Text feature dimension is: {}".format(text_dim))
    input_dims = (audio_dim, visual_dim, text_dim)

    # remove possible NaN values
    train_set.visual[train_set.visual != train_set.visual] = 0
    valid_set.visual[valid_set.visual != valid_set.visual] = 0
    test_set.visual[test_set.visual != test_set.visual] = 0

    train_set.audio[train_set.audio != train_set.audio] = 0
    valid_set.audio[valid_set.audio != valid_set.audio] = 0
    test_set.audio[test_set.audio != test_set.audio] = 0

    return train_set, valid_set, test_set, input_dims

def load_mosi(data_path):

    # parse the input args
    class MOSI(Dataset):
        '''
        PyTorch Dataset for MOSI, don't need to change this
        '''

        def __init__(self, audio, visual, text, labels):
            self.audio = audio
            self.visual = visual
            self.text = text
            self.labels = labels

        def __getitem__(self, idx):
            return [self.audio[idx, :], self.visual[idx, :], self.text[idx, :, :], self.labels[idx]]

        def __len__(self):
            return self.audio.shape[0]

    mosi_data = pickle.load(open(data_path + "mosi.pkl", 'rb'), encoding='latin1')
    mosi_train, mosi_valid, mosi_test = mosi_data['train'], mosi_data['valid'], mosi_data['test']

    train_audio, train_visual, train_text, train_labels \
        = mosi_train[AUDIO], mosi_train[VISUAL], mosi_train[TEXT], mosi_train[LABEL]
    valid_audio, valid_visual, valid_text, valid_labels \
        = mosi_valid[AUDIO], mosi_valid[VISUAL], mosi_valid[TEXT], mosi_valid[LABEL]
    test_audio, test_visual, test_text, test_labels \
        = mosi_test[AUDIO], mosi_test[VISUAL], mosi_test[TEXT], mosi_test[LABEL]

    train_audio = np.nan_to_num(train_audio)
    train_visual = np.nan_to_num(train_visual)
    train_text = np.nan_to_num(train_text)
    train_labels = np.nan_to_num(train_labels)
    valid_audio = np.nan_to_num(valid_audio)
    valid_visual = np.nan_to_num(valid_visual)
    valid_text = np.nan_to_num(valid_text)
    valid_labels = np.nan_to_num(valid_labels)
    test_audio = np.nan_to_num(test_audio)
    test_visual = np.nan_to_num(test_visual)
    test_text = np.nan_to_num(test_text)
    test_labels = np.nan_to_num(test_labels)

    print(train_audio.shape)
    print(train_visual.shape)
    print(train_text.shape)
    print(train_labels.shape)

    # code that instantiates the Dataset objects
    train_set = MOSI(train_audio, train_visual, train_text, train_labels)
    valid_set = MOSI(valid_audio, valid_visual, valid_text, valid_labels)
    test_set = MOSI(test_audio, test_visual, test_text, test_labels)



    audio_dim = train_set[0][0].shape[0]
    print("Audio feature dimension is: {}".format(audio_dim))
    visual_dim = train_set[0][1].shape[0]
    print("Visual feature dimension is: {}".format(visual_dim))
    text_dim = train_set[0][2].shape[1]
    print("Text feature dimension is: {}".format(text_dim))
    input_dims = (audio_dim, visual_dim, text_dim)

    # remove possible NaN values
    train_set.visual[train_set.visual != train_set.visual] = 0
    valid_set.visual[valid_set.visual != valid_set.visual] = 0
    test_set.visual[test_set.visual != test_set.visual] = 0

    train_set.audio[train_set.audio != train_set.audio] = 0
    valid_set.audio[valid_set.audio != valid_set.audio] = 0
    test_set.audio[test_set.audio != test_set.audio] = 0


    return train_set, valid_set, test_set, input_dims
