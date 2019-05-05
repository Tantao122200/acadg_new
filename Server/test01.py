import tensorflow as tf
# 创建2个矩阵，前者1行2列，后者2行1列，然后矩阵相乘：
matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2], [2]])
product = tf.matmul(matrix1,matrix2)

# 上边的操作是定义图，然后用会话Session去计算：
with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)