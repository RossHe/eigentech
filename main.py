from read_data_set import get_data
import numpy as np
import tensorflow as tf
import scipy.misc

x_train, y_train, x_test,y_test = get_data()
x_train,x_test = x_train.reshape(-1,28,28,1), x_test.reshape(-1,28,28,1)

def horizontalFlip(x):
    x = x[:,:,::-1,:]
    return x
x_flip = horizontalFlip(x_train)
y_flip = y_train
x_train = np.vstack((x_train,x_flip))
y_train = np.hstack((y_train,y_flip))
mask = np.arange(len(x_train))
np.random.shuffle(mask)
x_train = x_train[mask]
y_train = y_train[mask]

def preProcess(x):
    x = x[:] / 255
    return x

x_train = preProcess(x_train)
x_test = preProcess(x_test)

def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

y_train = dense_to_one_hot(y_train,10)
y_test = dense_to_one_hot(y_test,10)

x_val, y_val = x_train[55000:], y_train[55000:]
x_train, y_train = x_train[:55000], y_train[:55000]

def weight_variable(shape):
    init = tf.truncated_normal(shape,stddev = 0.1)
    return tf.Variable(init)

def bias_variable(shape):
    return tf.Variable(tf.zeros(shape))

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(h):
    return tf.nn.max_pool(h,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

x = tf.placeholder(tf.float32,[None,28,28,1])
y_ = tf.placeholder(tf.float32,[None,10])
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x,W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob)
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_dropout,W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(\
        tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv))
train_step = tf.train.AdamOptimizer(2e-5).minimize(cross_entropy)

correct_pred = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

batch_size = 256
for i in range(6001):
#    mask = np.random.shuffle(range(len(x_train)))
    start = (i * batch_size) % x_train.shape[0]
    end = min(start + batch_size, x_train.shape[0])
    batch_x, batch_y = x_train[start:end], y_train[start:end]
    if i % 100 == 0:
        train_loss = cross_entropy.eval(feed_dict={x:batch_x,y_:batch_y,keep_prob:0.5})
        val_loss = cross_entropy.eval(feed_dict={x:x_val,y_:y_val,keep_prob:1.0})
        train_acc = acc.eval(feed_dict={x:batch_x,y_:batch_y,keep_prob:0.5})
        val_acc = acc.eval(feed_dict={x:x_val,y_:y_val,keep_prob:1.0})
        print('iter: %d, train loss: %f, train acc: %f, val loss: %f, val acc: %f'\
              %(i,train_loss,train_acc,val_loss,val_acc))
    sess.run(train_step,feed_dict={x:batch_x,y_:batch_y,keep_prob:0.5})

loss = cross_entropy.eval(feed_dict={x:x_test,y_:y_test,keep_prob:1.0})
test_acc = acc.eval(feed_dict = {x:x_test,y_:y_test,keep_prob:1.0})
#acc2 = acc.eval(feed_dict = {x:x_test[len(x_test)//2:],y_:y_test[len(y_test)//2:],keep_prob:1.0})
print('test loss : %f ,test acc: %f'%(loss,test_acc))
sess.close()
