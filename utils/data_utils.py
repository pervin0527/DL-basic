import os
import wget
import gzip
import pickle
import struct
import tarfile
import numpy as np
import urllib.request


def download_cifar10(path):
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


def get_cifar10(data_dir):
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


def download_and_extract_mnist_data(download_path):
    # 이미 다운로드된 경우 스킵
    if os.path.exists(os.path.join(download_path, 'train-images-idx3-ubyte')) and \
       os.path.exists(os.path.join(download_path, 'train-labels-idx1-ubyte')) and \
       os.path.exists(os.path.join(download_path, 't10k-images-idx3-ubyte')) and \
       os.path.exists(os.path.join(download_path, 't10k-labels-idx1-ubyte')):
        print("MNIST data is already downloaded and extracted.")
        train_x = read_mnist_images(os.path.join(download_path, 'train-images-idx3-ubyte'))
        train_y = read_mnist_labels(os.path.join(download_path, 'train-labels-idx1-ubyte'))
        test_x = read_mnist_images(os.path.join(download_path, 't10k-images-idx3-ubyte'))
        test_y = read_mnist_labels(os.path.join(download_path, 't10k-labels-idx1-ubyte'))
        
        return train_x, train_y, test_x, test_y

    # MNIST 데이터 다운로드 경로
    mnist_urls = [
        "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
    ]
    
    # 다운로드 디렉토리 생성
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    
    # 각 파일 다운로드 및 압축 해제
    for url in mnist_urls:
        file_name = url.split("/")[-1]
        file_path = os.path.join(download_path, file_name)
        if not os.path.exists(file_path):
            print(f"Downloading {file_name}...")
            urllib.request.urlretrieve(url, file_path)
        
        # 압축 해제
        with gzip.open(file_path, 'rb') as f_in:
            with open(file_path[:-3], 'wb') as f_out:
                f_out.write(f_in.read())
        os.remove(file_path)
        print(f"Extracted {file_name}")
    
    # 데이터 읽기
    train_x = read_mnist_images(os.path.join(download_path, 'train-images-idx3-ubyte'))
    train_y = read_mnist_labels(os.path.join(download_path, 'train-labels-idx1-ubyte'))
    test_x = read_mnist_images(os.path.join(download_path, 't10k-images-idx3-ubyte'))
    test_y = read_mnist_labels(os.path.join(download_path, 't10k-labels-idx1-ubyte'))
    
    return train_x, train_y, test_x, test_y
    
def read_mnist_images(file_path):
    with open(file_path, 'rb') as f:
        magic, num_images, num_rows, num_cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, "Not a valid MNIST image file"
        data = np.fromfile(f, dtype=np.uint8)
        data = data.reshape(num_images, num_rows, num_cols)
    return data

def read_mnist_labels(file_path):
    with open(file_path, 'rb') as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        assert magic == 2049, "Not a valid MNIST label file"
        labels = np.fromfile(f, dtype=np.uint8)
    return labels


def make_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)
        print(f"{path} folder maded")
    else:
        print(f"{path} is already exist.")


def download_multi30k(save_path):
    URL = "https://github.com/multi30k/dataset/raw/master/data/task1/raw"
    FILES = ["test_2016_flickr.de.gz",
             "test_2016_flickr.en.gz",
             "train.de.gz",
             "train.en.gz",
             "val.de.gz",
             "val.en.gz"]
    
    save_path = f"{save_path}/Multi30k"
    make_dir(save_path)

    for file in FILES:
        file_name = file[:-3]
        if file_name == "test_2016_flickr.de_gz":
            file_name = "test.de"
        elif file_name == "test_2016_flickr.en.gz":
            file_name = "test.en"

        if os.path.exists(f"{save_path}/{file_name}"):
            pass
        else:
            url = f"{URL}/{file}"
            # print(f"{url}\n")

            wget.download(url, out=save_path)
            os.system(f"gzip -d {save_path}/{file}")
        
            if file == FILES[0]:
                os.system(f"cp {save_path}/{file[:-3]} {save_path}/test.de")
            elif file == FILES[1]:
                os.system(f"cp {save_path}/{file[:-3]} {save_path}/test.en")


def load_pickle(fname):
    with open(fname, "rb") as f:
        data = pickle.load(f)
    return data


def save_pickle(data, fname):
    with open(fname, "wb") as f:
        pickle.dump(data, f)


def make_cache(data_path):
    cache_path = f"{data_path}/cache"
    make_dir(cache_path)

    if not os.path.exists(f"{cache_path}/train.pkl"):
        for name in ["train", "val", "test"]:
            pkl_file_name = f"{cache_path}/{name}.pkl"

            with open(f"{data_path}/{name}.en", "r") as file:
                en = [text.rstrip() for text in file]
            
            with open(f"{data_path}/{name}.de", "r") as file:
                de = [text.rstrip() for text in file]
            
            data = [(en_text, de_text) for en_text, de_text in zip(en, de)]
            save_pickle(data, pkl_file_name)