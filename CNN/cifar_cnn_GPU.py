import tensorflow as tf
import argparse
import sys

from optimizer.Adam import AdamOptimizer
from optimizer.AMSGrad import AMSGradOptimizer
from optimizer.ACADG import ACADGOptimizer
from matplotlib import pyplot as plt

import pickle
import numpy as np
import pandas as pd

data_dir = r"./data/cifar-10-batches-py"


def train_data():
    train_data = {b'data': [], b'labels': []}
    for i in range(5):
        with open(data_dir + "/data_batch_" + str(i + 1), mode='rb') as file:
            data = pickle.load(file, encoding='bytes')
            train_data[b'data'] += list(data[b'data'])
            train_data[b'labels'] += data[b'labels']
    x_train = np.array(train_data[b'data']) / 255
    y_train = np.array(pd.get_dummies(train_data[b'labels']))
    return x_train, y_train


def test_data():
    test_data = {b'data': [], b'labels': []}
    with open(data_dir + "/test_batch", mode='rb') as file:
        data = pickle.load(file, encoding='bytes')
        test_data[b'data'] += list(data[b'data'])
        test_data[b'labels'] += data[b'labels']
    x_test = np.array(test_data[b'data']) / 255
    y_test = np.array(pd.get_dummies(test_data[b'labels']))
    return x_test, y_test


x_train, y_train = train_data()
x_test, y_test = test_data()


def next_train_batch(step=0, batch_size=128):
    batch_x = []
    batch_y = []
    for i in range(step * batch_size, (step + 1) * batch_size):
        batch_x.append(x_train[i % len(x_train)])
        batch_y.append(y_train[i % len(y_train)])
    batch_x = np.array(batch_x)
    batch_y = np.array(batch_y)
    return batch_x, batch_y


def next_test_batch(step=0, batch_size=128):
    batch_x = []
    batch_y = []
    for i in range(step * batch_size, (step + 1) * batch_size):
        batch_x.append(x_test[i % len(x_train)])
        batch_y.append(y_test[i % len(y_train)])
    batch_x = np.array(batch_x)
    batch_y = np.array(batch_y)
    return batch_x, batch_y


def load_data():
    x = tf.placeholder(tf.float32, [None, 3072])
    y_ = tf.placeholder(tf.float32, [None, 10])
    return x, y_


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.2)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding="SAME")


def cnn(input_tensor):
    # 第一层卷积
    W_conv1 = weight_variable([5, 5, 3, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(input_tensor, [-1, 3, 32, 32])
    x_image = tf.transpose(x_image, [0, 2, 3, 1])  # [-1, 32, 32, 3]
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # 第二层卷积
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # 全连接层
    W_fc1 = weight_variable([8 * 8 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # 输出层
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

    return y_conv


def loss_prediction(labels, logits):
    myloss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    return tf.reduce_mean(myloss)


def accuracy_prediction(labels, logits):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def optimization(name, cross_entropy):
    global_step = tf.Variable(tf.constant(1.0))
    learning_rate_decayed = FLAGS.learning_rate / tf.sqrt(global_step)
    if name == "ADAM":
        train_step = AdamOptimizer(learning_rate_decayed).minimize(cross_entropy, global_step=global_step)
    elif name == "AMSGRAD":
        train_step = AMSGradOptimizer(learning_rate_decayed).minimize(cross_entropy, global_step=global_step)
    elif name == "ACADG":
        train_step = ACADGOptimizer(learning_rate_decayed).minimize(cross_entropy, global_step=global_step)
    else:
        train_step = tf.train.GradientDescentOptimizer(learning_rate_decayed).minimize(cross_entropy,
                                                                                       global_step=global_step)
    return train_step


def iterations(input_tensor, labels, loss, accuracy, optimizate):
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    test_loss = []
    train_loss = []
    test_acc = []
    train_acc = []
    train_acc_data = 0
    test_acc_data = 0
    count = []
    for i in range(1,FLAGS.max_steps+1):
        batch_xs, batch_ys = next_train_batch(i-1)
        _, loss_train, acc_train = sess.run([optimizate, loss, accuracy],
                                            feed_dict={input_tensor: batch_xs, labels: batch_ys})
        if (i % 600 == 0):
            count.append(i)
            train_loss.append(loss_train)
            train_acc.append(acc_train)
            batch_xs, batch_ys = next_train_batch(i)
            test_loss_, test_acc_ = sess.run([loss, accuracy], feed_dict={input_tensor: batch_xs, labels: batch_ys})
            test_loss.append(test_loss_)
            test_acc.append(test_acc_)
            print("step is {},train loss is {},train acc is {},test loss is {},test acc is {}".format(i, loss_train,
                                                                                                      acc_train,
                                                                                                      test_loss_,
                                                                                                      test_acc_))
        if (i == FLAGS.max_steps):
            train_acc_data = acc_train
            test_acc_data = sess.run(accuracy, feed_dict={input_tensor: x_test, labels: y_test})
            print("模型测试：", end=" ")
            print(train_acc_data, end=" ")
            print(test_acc_data)
    return count, train_loss, test_loss, train_acc, test_acc, train_acc_data, test_acc_data

def show_train_loss(count, data, label):
    plt.figure(1)
    plt.ylabel("Train loss")
    plt.xlabel("Iterations")
    plt.plot(count, data, label=label)
    plt.legend()
    plt.savefig("./cifar_train_loss.png")


def show_train_acc(count, data, label):
    plt.figure(2)
    plt.ylabel("Train acc")
    plt.xlabel("Iterations")
    plt.plot(count, data, label=label)
    plt.legend()
    plt.savefig("./cifar_train_acc.png")


def show_test_loss(count, data, label):
    plt.figure(3)
    plt.ylabel("Test loss")
    plt.xlabel("Iterations")
    plt.plot(count, data, label=label)
    plt.legend()
    plt.savefig("./cifar_test_loss.png")

def show_test_acc(count, data, label):
    plt.figure(4)
    plt.ylabel("Test acc")
    plt.xlabel("Iterations")
    plt.plot(count, data, label=label)
    plt.legend()
    plt.savefig("./cifar_test_acc.png")

def main(_):
    x, y = load_data()
    y_l = cnn(x)
    loss = loss_prediction(y, y_l)
    accuracy = accuracy_prediction(y, y_l)

    myoptimization = ["ADAM", "AMSGRAD","ACADG"]
    a1 = []
    a2 = []
    for name in myoptimization:
        op = optimization(name, loss)
        print(name + "的train和test准确率: ")
        count, train_loss, test_loss, train_acc, test_acc, train_acc_data, test_acc_data = iterations(x, y, loss,
                                                                                                      accuracy, op)
        a1.append(train_acc_data)
        a2.append(test_acc_data)
        show_train_loss(count, train_loss, name)
        show_test_loss(count, test_loss, name)
        show_train_acc(count, train_acc, name)
        show_test_acc(count, test_acc, name)

    w_csv = pd.DataFrame({"name": myoptimization, "train_acc": a1, "test_acc": a2})
    w_csv.to_csv("./cifar_cnn_gpu.csv", encoding="utf-8")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_steps', type=int, default=300000,
                        help='Number of steps to run trainer.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
