# hw0
## Question 1: A basic add function, and testing/autograding basics

## Question 2: Loading MNIST data
```py
def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0 
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    # Read the labels file
    with gzip.open(label_filename, 'rb') as lbl_file:
        lbl_file.read(8)  # skip the magic number and the number of labels
        y = np.frombuffer(lbl_file.read(), dtype=np.uint8)

    # Read the images file
    with gzip.open(image_filename, 'rb') as img_file:
        img_file.read(16)  # skip the magic number, number of images, rows, and columns
        X = np.frombuffer(img_file.read(), dtype=np.uint8).astype(np.float32)
        X = X.reshape(-1, 28 * 28)  # reshape into (num_examples x 784)
        X /= 255.0  # normalize values to be in the range [0.0, 1.0]
    return X, y
    ### END YOUR CODE
```
## Question 3: Softmax loss
```py
def softmax_loss(Z, y):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.uint8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    
    print("Z shape: "+str(Z.shape))
    print("y shape: "+str(y.shape))
    
    # 一般为了方便，将维度先提取出来
    batch_size=Z.shape[0]
    # 这里索引注意一下下表不要写错
    z_y=Z[np.arange(batch_size),y]
    Z=np.exp(Z)
    Z=np.sum(Z,axis=1)
    Z=np.log(Z)
    res=np.sum(Z-z_y)/batch_size
    print(res)
    return res
```
## Question 4: Stochastic gradient descent for softmax regression
```py
def softmax_loss(Z, y):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.uint8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    ### BEGIN YOUR CODE
    print("Z shape: "+str(Z.shape))
    print("y shape: "+str(y.shape))
 
    batch_size=Z.shape[0]
    z_y=Z[np.arange(batch_size),y]
    Z=np.exp(Z)
    Z=np.sum(Z,axis=1)
    Z=np.log(Z)
    res=np.sum(Z-z_y)/batch_size
    print(res)
    return res


def softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size.  This function should modify the
    theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    total_item=X.shape[0]
    num_classes=theta.shape[1]
    for i in range(0,total_item,batch):
      inputs=X[i:i+batch]
      Z=np.exp(inputs @ theta)
      Z=Z/np.sum(Z,axis=1,keepdims=True)
      # Z:(batch, num_classes)
      gt=y[i:i+batch]
      # 这里数组索引需要注意
      I_y=np.eye(num_classes)[gt]
      # print(inputs.shape)
      # print(I_y.shape)
      # print(Z.shape)
      # 这里注意一下各种维度转换函数的区别
      grade=inputs.transpose(1,0) @ (Z-I_y)/batch
      # 参数更新注意正负号
      theta-=grade*lr
    return
```


## Question 5: SGD for a two-layer neural network
```py
def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).  It should modify the
    W1 and W2 matrices in place.
    
    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (np.ndarray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (np.ndarray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch
    
    Returns:
        None
    """
    num_examples = X.shape[0]
    input_dim = X.shape[1]
    hidden_dim = W1.shape[1]
    num_classes = W2.shape[1]

    #print(num_examples,input_dim,hidden_dim,num_classes)

    for i in range(0, num_examples,batch):
        batch_X=X[i:i+batch]
        batch_y=y[i:i+batch]
        # Z=relu(X W1)W2
        # Z=(batch,num_class)
        Z= np.maximum(0, batch_X @ W1) @ W2

        Z_norm=np.exp(Z)/np.sum(np.exp(Z),axis=1,keepdims=True)
        e_y=np.eye(num_classes)[batch_y]
        plpz=Z_norm-e_y
        # a:(batch,num_classes)
        # partial_h/partial_w2:(batch, hid_dim)
        # partial_h/partial_w1:
        pzpw2=np.maximum(batch_X @ W1,0)
        # pzpw2:(batch_size,hidden_dim)
        pw2=pzpw2.T @ plpz
        
        pw1=batch_X.T @ ((pzpw2>0).astype(int)*(plpz @ W2.T))

        pw1/=batch
        pw2/=batch

        W1-=pw1*lr
        W2-=pw2*lr
```
## Question 6: Softmax regression in C++
```cpp

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
                                  float *theta, size_t m, size_t n, size_t k,
                                  float lr, size_t batch)
{
    size_t num_examples = m;
    size_t num_classes = k;
    size_t input_dim = n;

    auto mat_mul = [&](const float* X, const float* Y, float *res, size_t a, size_t b, size_t c) {
        // X.shape=(a,b) Y.shape=(b,c)
        std::memset(res, 0, sizeof(float)*a*c);
        for (size_t i = 0; i < a; i++) {
            for (size_t j = 0; j < c; j++) {
                for (size_t l = 0; l < b; l++) {
                    res[i*c + j] += X[i*b + l] * Y[l*c + j];
                }
            }
        }
    };

    auto mat_exp = [&](float* X, size_t a, size_t b) {
        for (size_t i = 0; i < a; i++) {
            for (size_t j = 0; j < b; j++) {
                X[i*b + j] = std::exp(X[i*b + j]);
            }
        }
    };

    auto mat_sum_row = [&](const float* X, size_t a, size_t b, float* sums) {
        for (size_t i = 0; i < a; i++) {
            sums[i] = 0;
            for (size_t j = 0; j < b; j++) {
                sums[i] += X[i*b + j];
            }
        }
    };

    auto mat_div_row = [&](float* X, const float* sums, size_t a, size_t b) {
        for (size_t i = 0; i < a; i++) {
            for (size_t j = 0; j < b; j++) {
                X[i*b + j] /= sums[i];
            }
        }
    };

    auto mat_add = [&](float* X, const float *Y, size_t a, size_t b) {
        for (size_t i = 0; i < a; i++) {
            for (size_t j = 0; j < b; j++) {
                X[i*b + j] += Y[i*b + j];
            }
        }
    };

    auto mul = [&](float* X, float op, size_t a, size_t b) {
        for (size_t i = 0; i < a; i++) {
            for (size_t j = 0; j < b; j++) {
                X[i*b + j] *= op;
            }
        }
    };
    auto mat_div = [&](float* X, float op, size_t a, size_t b) {
        for (size_t i = 0; i < a; i++) {
            for (size_t j = 0; j < b; j++) {
                X[i*b + j] /= op;
            }
        }
    };

    auto T = [&](const float *X, float *res, size_t a, size_t b) {
        for (size_t i = 0; i < a; i++) {
            for (size_t j = 0; j < b; j++) {
                res[j*a + i] = X[i*b + j];
            }
        }
    };

    float *logits = new float[batch * num_classes];
    float *I_y = new float[batch * num_classes];
    float *delta = new float[input_dim * num_classes];
    float *transposed_batch_x = new float[input_dim * batch];
    float *row_sums = new float[batch];

    for (size_t i = 0; i < num_examples; i += batch) {
        size_t current_batch = (i + batch > num_examples) ? num_examples - i : batch;
        const float *batch_x = X + i * input_dim;
        const unsigned char *batch_y = y + i;

        mat_mul(batch_x, theta, logits, current_batch, input_dim, num_classes);
        mat_exp(logits, current_batch, num_classes);
        mat_sum_row(logits, current_batch, num_classes, row_sums);
        mat_div_row(logits, row_sums, current_batch, num_classes);

        std::memset(I_y, 0, sizeof(float) * batch * num_classes);
        for (size_t j = 0; j < current_batch; j++) {
            I_y[j*num_classes + batch_y[j]] = 1;
        }

        T(batch_x, transposed_batch_x, current_batch, input_dim);

        mul(I_y,-1,current_batch,num_classes);

        mat_add(logits, I_y, current_batch, num_classes);

        mat_mul(transposed_batch_x, logits, delta, input_dim, current_batch, num_classes);

        mat_div(delta, static_cast<float>(current_batch), input_dim, num_classes);

        mul(delta, -lr, input_dim, num_classes);

        mat_add(theta, delta, input_dim, num_classes);
    }

    delete[] logits;
    delete[] I_y;
    delete[] delta;
    delete[] transposed_batch_x;
    delete[] row_sums;
}
```