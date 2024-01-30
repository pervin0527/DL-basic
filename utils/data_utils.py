import os
import wget
import pickle
import tarfile
import numpy as np


def download_and_extract(path):
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    file_name = os.path.join(path, url.split('/')[-1])

    # 파일이 이미 존재하는지 확인
    if not os.path.exists(file_name):
        # 파일 다운로드
        wget.download(url, file_name)
        print("\n파일 다운로드 완료.")
    else:
        print("\n파일이 이미 존재합니다.")

    # 압축 해제 (파일이 존재하지 않고, .tar.gz 확장자일 때만)
    if not os.path.exists(file_name.replace('.tar.gz', '')) and file_name.endswith(".tar.gz"):
        with tarfile.open(file_name, "r:gz") as tar:
            tar.extractall(path=path)
        print("압축 해제 완료.")
    else:
        print("압축 해제 과정을 건너뜁니다.")


def get_dataset(data_dir):
    train_data_files = ["data_batch_1",
                        "data_batch_2",
                        "data_batch_3",
                        "data_batch_4",
                        "data_batch_5"]

    train_data_files = [f"{data_dir}/{file}" for file in train_data_files]
    test_data_file = [f"{data_dir}/test_batch"]

    return train_data_files, test_data_file


def unpickle(files):
    dataset = {'data': [], 'labels': []}
    for file in files:
        with open(file, 'rb') as fo:
            data_dict = pickle.load(fo, encoding='bytes')
            images = data_dict[b'data']
            labels = data_dict[b'labels']
            dataset['data'].extend(images)
            dataset['labels'].extend(labels)
    dataset['data'] = np.array(dataset['data'])
    dataset['labels'] = np.array(dataset['labels'])
    
    return dataset


def vec2img(images):
    X = []
    for image in images:
        ## 처음 1024개의 요소가 빨간색 채널, 다음 1024개의 요소가 녹색 채널, 마지막 1024개의 요소가 파란색 채널.
        image = np.reshape(image, (3, 32, 32)) # 이미지를 (3, 32, 32)로 재구성
        image = np.transpose(image, (1, 2, 0)) # # 축을 재배열하여 (32, 32, 3)으로 변환
        image = image.astype(np.uint8)
        X.append(image)

    return np.array(X)


def cat_filter(labels, target=3):
    Y = []
    for label in labels:
        if label == target:
            Y.append([1])
        else:
            Y.append([0])

    return np.array(Y).transpose((1, 0))


if __name__ == "__main__":
    download_and_extract("/home/pervinco/Datasets/test")