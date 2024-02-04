import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def softmax(z):
    e_z = np.exp(z - np.max(z))
    return e_z / e_z.sum(axis=0, keepdims=True)


def initialize_parameters_random(layers_dims):
    """
    Normal Distribution을 기반으로 랜덤값을 선정.

    Arguments:
     - layer_dims : 각 layer에 설정할 unit의 수. list 타입
    
    Returns:
     - parameters : 각 layer에 설정되는 weight(matrix, "W1", "W2",..."WL"), bias(vector, "b1", "b2",..."bL")의 dictionary 타입
                    W1 : weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 : bias vector of shape (layers_dims[1], 1)
                    ...
                    WL : weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL : bias vector of shape (layers_dims[L], 1)
    """
    np.random.seed(3)

    parameters = {}
    L = len(layers_dims)
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * 10
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
        
    return parameters


def initialize_parameters_zeros(layers_dims):
    ## zero initialization
    
    parameters = {}
    L = len(layers_dims) # number of layers in the network
    
    for l in range(1, L):
        parameters["W" + str(l)] = np.zeros((layers_dims[l], layers_dims[l-1]))
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))
                
    return parameters


def initialize_parameters_he(layers_dims):  
    ## He Normal Distribution을 따라 확률 변수들에 확률을 부여하고, 랜덤으로 가중치 값을 선정.
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims) - 1
     
    for l in range(1, L + 1):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(2 / layers_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
                
    return parameters


def forward_propagation(X, parameters):
    """
    Arguments:
     - X : input data (n_x, m) n_x는 feature 수, m은 데이터 샘플 수
     - parameters : 각 layer의 trainable params dict타입 {"W1", "b1", ..., "WL", "bL"}

    Returns:
     - AL : 출력층(L번째 layer)의 activation feature
     - caches : 각 층의 (input, weight, bias, output)
    """
    caches = []
    A = X
    L = len(parameters) // 2
    
    # [LINEAR -> RELU]*(L-1)
    for l in range(1, L):
        A_prev = A 
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]
        Z = np.dot(W, A_prev) + b
        A = np.maximum(0, Z)  # ReLU
        cache = (A_prev, W, b, Z)
        caches.append(cache)
        
    # LINEAR -> SIGMOID
    WL = parameters['W' + str(L)]
    bL = parameters['b' + str(L)]
    ZL = np.dot(WL, A) + bL
    AL = 1 / (1 + np.exp(-ZL))  # Sigmoid
    cache = (A, WL, bL, ZL)
    caches.append(cache)
    
    assert(AL.shape == (1, X.shape[1]))
    
    return AL, caches


def compute_cost(AL, Y):
    """
    Arguments:
     - AL : label 예측에 해당하는 확률 벡터. (1, num of data samples)
     - Y : ground-truth 벡터. (1, num of data samples)

    Returns:
     - cost : cross-entropy cost
    """
    m = Y.shape[1]

    cost = (-1/m) * np.sum(Y * np.log(AL) + (1- Y) * np.log(1 - AL))
    cost = np.squeeze(cost)

    return cost


def compute_cost_with_regularization(A3, Y, parameters, lambd):
    """
    L2 Regularization을 적용한 cost function.
    Arguments:
     - A3 : L번째 layer(출력층)의 출력값 (output size, number of examples)
     - Y : ground-truth (output size, number of examples)
     - parameters : 각 층에 적용된 trainable parameters
    
    Returns:
     - cost : L2 정규화가 더해진 cost값.
    """
    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    
    cross_entropy_cost = compute_cost(A3, Y)
    L2_regularization_cost = (lambd / (2 * m)) * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)))
    
    cost = cross_entropy_cost + L2_regularization_cost
    
    return cost


def forward_propagation_with_dropout(X, parameters, keep_prob = 0.5):
    """
    LINEAR -> RELU + DROPOUT -> LINEAR -> RELU + DROPOUT -> LINEAR -> SIGMOID.
    
    Arguments:
     - X : input dataset. (num features, number of examples)
     - parameters : 각 layer에 적용되는 trainable parameters.
                    W1 : weight matrix of shape (20, 2)
                    b1 : bias vector of shape (20, 1)
                    W2 : weight matrix of shape (3, 20)
                    b2 : bias vector of shape (3, 1)
                    W3 : weight matrix of shape (1, 3)
                    b3 : bias vector of shape (1, 1)
     - keep_prob : dropout이 적용되지 않는, 활성화되는 unit의 수(백분율)
    
    Returns:
     - A3 : output layer의 출력.
     - cache : 각 layer의 (input, weight, bias, output)
    """
    
    np.random.seed(1)
    
    # retrieve parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1, X) + b1
    A1 = np.maximum(0, Z1) ## ReLU

    D1 = np.random.rand(A1.shape[0], A1.shape[1])
    D1 = (D1 < keep_prob).astype(int)
    A1 = np.multiply(D1, A1)
    A1 /= keep_prob
    
    Z2 = np.dot(W2, A1) + b2
    A2 = np.maximum(0, Z2) ## ReLU

    D2 = np.random.rand(A2.shape[0], A2.shape[1])
    D2 = (D2 < keep_prob).astype(int)
    A2 = np.multiply(D2, A2)
    A2 /= keep_prob
    
    
    Z3 = np.dot(W3, A2) + b3
    A3 = 1 / (1 + np.exp(-Z3)) ## Sigmoid
    
    cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)
    
    return A3, cache