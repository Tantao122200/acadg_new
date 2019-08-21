import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf
from matplotlib import pyplot as plt
import argparse
import sys
import pandas as pd
from optimizer.Adam import AdamOptimizer
from optimizer.AMSGrad import AMSGradOptimizer
from optimizer.ACADG import ACADGOptimizer


def load_data():
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    return mnist, x, y_


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding="SAME")


def cnn(input_tensor):
    # 第一层卷积
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(input_tensor, [-1, 28, 28, 1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # 第二层卷积
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # 全连接层
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
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


def iterations(mnist, input_tensor, labels, loss, accuracy, optimizate):
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
    for i in range(FLAGS.max_steps):
        batch_xs, batch_ys = mnist.train.next_batch(128)
        _, loss_train, acc_train = sess.run([optimizate, loss, accuracy],
                                            feed_dict={input_tensor: batch_xs, labels: batch_ys})
        if (i % 100 == 0):
            count.append(i)
            train_loss.append(loss_train)
            train_acc.append(acc_train)
            batch_xs, batch_ys = mnist.test.next_batch(128)
            test_loss_, test_acc_ = sess.run([loss, accuracy], feed_dict={input_tensor: batch_xs, labels: batch_ys})
            test_loss.append(test_loss_)
            test_acc.append(test_acc_)
        if (i == FLAGS.max_steps-1):
            batch_xs, batch_ys = mnist.train.next_batch(128)
            train_acc_data = sess.run(accuracy,
                                      feed_dict={input_tensor: batch_xs, labels: batch_ys})
            batch_xs, batch_ys = mnist.test.next_batch(128)
            test_acc_data = sess.run(accuracy, feed_dict={input_tensor: batch_xs, labels: batch_ys})
            print(train_acc_data)
            print(test_acc_data)
    return count, train_loss, test_loss, train_acc, test_acc, train_acc_data, test_acc_data


def show_train_loss(count, data, label):
    plt.figure(1)
    plt.ylabel("Train loss")
    plt.xlabel("Iterations")
    plt.plot(count, data, label=label)
    plt.legend()
    plt.savefig("./mnist_train_loss.png")


def show_train_acc(count, data, label):
    plt.figure(2)
    plt.ylabel("Train acc")
    plt.xlabel("Iterations")
    plt.plot(count, data, label=label)
    plt.legend()
    plt.savefig("./mnist_train_acc.png")


def show_test_loss(count, data, label):
    plt.figure(3)
    plt.ylabel("Test loss")
    plt.xlabel("Iterations")
    plt.plot(count, data, label=label)
    plt.legend()
    plt.savefig("./mnist_test_loss.png")

def show_test_acc(count, data, label):
    plt.figure(4)
    plt.ylabel("Test acc")
    plt.xlabel("Iterations")
    plt.plot(count, data, label=label)
    plt.legend()
    plt.savefig("./mnist_test_acc.png")


def main(_):
    mnist, x, y = load_data()
    y_l = cnn(x)
    loss = loss_prediction(y, y_l)
    accuracy = accuracy_prediction(y, y_l)

    myoptimization = ["ADAM", "AMSGRAD", "ACADG"]
    a1 = []
    a2 = []
    for name in myoptimization:
        op = optimization(name, loss)
        print(name + "的train和test准确率: ")
        count, train_loss, test_loss, train_acc, test_acc, train_acc_data, test_acc_data = iterations(mnist, x, y, loss,
                                                                                                      accuracy, op)
        a1.append(train_acc_data)
        a2.append(test_acc_data)
        # show_train_loss(count, train_loss, name)
        # show_test_loss(count, test_loss, name)
        # show_train_acc(count, train_acc, name)
        # show_test_acc(count, test_acc, name)

    w_csv = pd.DataFrame({"name": myoptimization, "train_acc": a1, "test_acc": a2})
    w_csv.to_csv("./cnn_mnist.csv", encoding="utf-8")

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_steps', type=int, default=50000,
                        help='Number of steps to run trainer.')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Initial learning rate')
    parser.add_argument('--data_dir', type=str, default=r"./data/mnist",
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
