#Author:xiao luo
#Author:xiao luo
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
batch_size = 64
max_step =2000
mnist = input_data.read_data_sets("I:\mnist",one_hot=True)
x = tf.placeholder(dtype=tf.float32,shape=[None,784])
y = tf.placeholder(dtype=tf.float32,shape=[None,10])

w = tf.Variable(tf.truncated_normal(shape=[784,10],stddev=0.01))
b = tf.Variable(tf.truncated_normal(shape=[10],stddev=0.01))

y_ = tf.nn.softmax(tf.matmul(x,w)+b)
loss = tf.reduce_sum(tf.square(y-y_))
#cross_entropy = -tf.reduce_sum(y*tf.log(y_))
train = tf.train.GradientDescentOptimizer(0.005).minimize(loss)#没有隐藏层时，建议0.01注意学习率太大会导致loss不下降
acc1 = tf.equal(tf.argmax(y,axis=1),tf.argmax(y_,axis=1))
acc = tf.reduce_mean(tf.cast(acc1,tf.float32))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(max_step):
        batch_x,batch_y = mnist.train.next_batch(batch_size)
        feed_dict = {x:batch_x,y:batch_y}
        sess.run(train,feed_dict=feed_dict)

        if i%100==0:
            print("steap:%d acc:%f"%(i,sess.run(acc,feed_dict={x:mnist.test.images,y:mnist.test.labels})))
            #print("loss:%f",sess.run(loss,feed_dict))