import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from dataset import Multi30kDataset

if __name__ == "__main__":
    train_dataset, valid_dataset, test_dataset = Multi30kDataset(max_length=1000, lower=True, min_freq=2).get_datasets()
    print(train_dataset[0])