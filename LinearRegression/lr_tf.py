import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
data=pd.read_csv('kc_house_data.csv')
x_data=np.array(data.iloc[:,3:])
y_data=np.array(data.iloc[:,2:3])
x_data=np.array(data.iloc[:,3:])
x_data=(x_data-np.mean(x_data))/np.std(x_data)
y_data=np.array(data.iloc[:,2:3])
train_x,test_x,train_y,test_y=train_test_split(x_data,y_data,test_size=0.2)

lr=0.001
display_step = 50
epochs=10000
X=tf.placeholder("float64")
Y=tf.placeholder("float64")

W=tf.Variable(np.random.rand(18),name="weight",dtype=np.float64)
b=tf.Variable(np.random.rand(),name="bias",dtype=np.float64)


pred = tf.add(tf.multiply(X, W), b)

cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*x_data.shape[0])
optimizer = tf.train.GradientDescentOptimizer(lr).minimize(cost)

init =tf.global_variables_initializer()

with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Fit all training data
    for epoch in range(epochs):
        sess.run(optimizer, feed_dict={X: train_x, Y: train_y})

        if (epoch + 1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_x, Y: train_y})
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c),
                  "W=", sess.run(W), "b=", sess.run(b))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: x_data, Y:y_data})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')
    w=W.eval(sess)
    b=b.eval(sess)

    # Graphic display
    # plt.plot(x_data, y_data, 'ro', label='Original data')
    # plt.plot(x_data, sess.run(W) * x_data + sess.run(b), label='Fitted line')
    # plt.legend()
    # plt.show()

v = np.zeros((17290))
for i in range(17290):
    v[i] = i
plt.scatter(v, train_y, color='g')
plt.scatter(v,np.add(np.dot(train_x,w),b), color='r')
plt.show()