import random
from collections import defaultdict
from os import listdir
from os.path import isfile, join, isdir

import soundfile as sf
from librosa.feature import mfcc
from sklearn.neighbors import KNeighborsClassifier

DATA_DIR_PATH = 'Imiona'


def extract_features(example_file):
    soundfile, samplerate = sf.read(example_file)
    return mfcc(y=soundfile, sr=samplerate, S=None, n_mfcc=13, dct_type=2, n_fft=1024, hop_length=64).T


def most_frequent_item(items):
    items_by_key = defaultdict(lambda: [])
    for item in items:
        items_by_key[item].append(item)

    most_frequent = None
    max_frequency = 0
    for values_arr in items_by_key.values():
        if len(values_arr) > max_frequency:
            most_frequent = values_arr[0]
            max_frequency = len(values_arr)

    return most_frequent


def extract_label_from_filename(filename):
    return filename[:2] + filename[3:4] + filename[5:-7]


def load_data():
    labels_with_dirs = [join(DATA_DIR_PATH, f) for f in listdir(DATA_DIR_PATH) if isdir(join(DATA_DIR_PATH, f))]
    files = []
    for (directory) in labels_with_dirs:
        examples_of_label = [(extract_label_from_filename(f), join(directory, f)) for f in listdir(directory) if
                             isfile(join(directory, f))]
        files.extend(examples_of_label)

    return files


def split_data(to_split, test_fraction=0.2):
    test_set_size = int(len(to_split) * test_fraction)
    test_set = []
    for i in range(test_set_size):
        test_item_index = random.randint(0, test_set_size - i)
        test_set.append(to_split[test_item_index])
        to_split[-1], to_split[test_item_index] = to_split[test_item_index], to_split[-1]

    return to_split[:-test_set_size], test_set


if __name__ == "__main__":
    data = load_data()
    train, test = split_data(data, test_fraction=0.5)
    print("Full dataset size is {}. Train set has size {}. Test set has size {}"
          .format(len(data), len(train), len(test)))

    x_train = []
    y_train = []
    for (label, file) in train:
        features_list = extract_features(file)
        for features in features_list:
            x_train.append(features)
            y_train.append(label)

    unique_labels_count = len(set(y_train))

    nbrs = KNeighborsClassifier(n_neighbors=unique_labels_count).fit(x_train, y_train)

    correct = 0
    incorrect = 0

    for (label, file) in test:
        predictions = nbrs.predict(extract_features(file))
        y_pred = most_frequent_item(predictions)
        if y_pred == label:
            correct += 1
        else:
            print (label, y_pred)
            incorrect += 1

    print('correct:', correct)
    print('incorrect:', incorrect)
