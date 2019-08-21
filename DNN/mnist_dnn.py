import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf
from matplotlib import pyplot as plt
import argparse
import sys
import pandas as pd
import time

from optimizer.Adam import AdamOptimizer
from optimizer.AMSGrad import AMSGradOptimizer
from optimizer.ACADG import ACADGOptimizer


def load_data():
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)
    return mnist, x, y_, keep_prob


def nn_layer(input_tensor, input_dim, output_dim, keep_prob, act=tf.nn.relu):
    weights = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1))
    biases = tf.Variable(tf.constant(0.1, shape=[output_dim]))
    preactivate = tf.matmul(input_tensor, weights) + biases
    activations = act(preactivate, name='activation')
    activations = tf.nn.dropout(activations, keep_prob)
    return activations


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


def iterations(name, mnist, input_tensor, labels, keep_prob, loss, accuracy, optimizate, start_time):
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
    mytime = []
    prob_train = FLAGS.dropout
    prob_test = 1.0
    for i in range(FLAGS.max_steps):
        batch_xs, batch_ys = mnist.train.next_batch(128)
        _, loss_train, acc_train = sess.run([optimizate, loss, accuracy],
                                            feed_dict={input_tensor: batch_xs, labels: batch_ys, keep_prob: prob_train})
        if (i % 100 == 0):
            count.append(i)
            train_loss.append(loss_train)
            train_acc.append(acc_train)
            loss_test, acc_test = sess.run([loss, accuracy],
                                           feed_dict={input_tensor: mnist.test.images, labels: mnist.test.labels,
                                                      keep_prob: prob_test})
            test_loss.append(loss_test)
            test_acc.append(acc_test)
            mytime.append(time.time() - start_time)
            print(
                "After %.5f time,g_step is %d,train_loss is %.5f,train_acc is %.5f,test_loss is %.5f,test_acc is %.5f" % (
                time.time() - start_time, i, loss_train, acc_train,loss_test, acc_test))
        if (i == FLAGS.max_steps - 1):
            train_acc_data = sess.run(accuracy,
                                      feed_dict={input_tensor: mnist.train.images, labels: mnist.train.labels,
                                                 keep_prob: prob_test})
            test_acc_data = sess.run(accuracy, feed_dict={input_tensor: mnist.test.images, labels: mnist.test.labels,
                                                          keep_prob: prob_test})
            print(name + "的train和test准确率：", end="")
            print(train_acc_data, end=" ")
            print(test_acc_data)
    return count, train_loss, test_loss, train_acc, test_acc, train_acc_data, test_acc_data,mytime


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

def show_train_loss_time(time, data, label):
    plt.figure(5)
    plt.ylabel("Train loss")
    plt.xlabel("Time")
    plt.plot(time, data, label=label)
    plt.legend()
    plt.savefig("./mnist_train_loss_time.png")


def show_train_acc_time(time, data, label):
    plt.figure(6)
    plt.ylabel("Train acc")
    plt.xlabel("Time")
    plt.plot(time, data, label=label)
    plt.legend()
    plt.savefig("./mnist_train_acc_time.png")


def show_test_loss_time(time, data, label):
    plt.figure(7)
    plt.ylabel("Test loss")
    plt.xlabel("Time")
    plt.plot(time, data, label=label)
    plt.legend()
    plt.savefig("./mnist_test_loss_time.png")


def show_test_acc_time(time, data, label):
    plt.figure(8)
    plt.ylabel("Test acc")
    plt.xlabel("Time")
    plt.plot(time, data, label=label)
    plt.legend()
    plt.savefig("./mnist_test_acc_time.png")


def main(_):
    mnist, x, y, keep_prob = load_data()
    hiddle = nn_layer(x, 784, 100, keep_prob)
    y_l = nn_layer(hiddle, 100, 10, keep_prob, tf.identity)
    loss = loss_prediction(y, y_l)
    accuracy = accuracy_prediction(y, y_l)

    myoptimization = ["ADAM", "AMSGRAD", "ACADG"]
    a1 = []
    a2 = []
    for name in myoptimization:
        start_time = time.time()
        op = optimization(name, loss)
        # print(name + "的train和test准确率：", end="")
        count, train_loss, test_loss, train_acc, test_acc, train_acc_data, test_acc_data,mytime = iterations(name, mnist, x, y,
                                                                                                      keep_prob, loss,
                                                                                                      accuracy, op,
                                                                                                      start_time)
        a1.append(train_acc_data)
        a2.append(test_acc_data)
        show_train_loss(count, train_loss, name)
        show_test_loss(count, test_loss, name)
        show_train_acc(count, train_acc, name)
        show_test_acc(count, test_acc, name)

    w_csv = pd.DataFrame({"name": myoptimization, "train_acc": a1, "test_acc": a2})
    w_csv.to_csv("./dnn_mnist.csv", encoding="utf-8")

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_steps', type=int, default=100000,
                        help='Number of steps to run trainer.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=1.0,
                        help='Keep probability for training dropout.')
    parser.add_argument('--data_dir', type=str, default=r"C:\tantao\pycharm\workspace\acadg_new\MNIST_data",
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
