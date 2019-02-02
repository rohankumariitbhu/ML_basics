import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
rng = np.random
import pandas as pd
# Parameters
lr = 0.001
epochs = 100000
display_step = 50
train_data=pd.read_csv('train.csv')
test_data=pd.read_csv('test.csv')
x_data=np.array(train_data.iloc[:,0:1])
x_data=(x_data-np.mean(x_data))/np.std(x_data)
y_data=np.array(train_data.iloc[:,1:])
xtest_data=np.array(test_data.iloc[:,0:1])
ytest_data=np.array(test_data.iloc[:,1:])

X = tf.placeholder("float")
Y = tf.placeholder("float")

W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

pred = tf.add(tf.multiply(X, W), b)

cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*x_data.shape[0])
optimizer = tf.train.GradientDescentOptimizer(lr).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()


with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Fit all training data
    for epoch in range(epochs):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})


        if (epoch + 1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: x_data, Y: y_data})
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c),
                  "W=", sess.run(W), "b=", sess.run(b))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: x_data, Y:y_data})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    # Graphic display
    plt.plot(x_data, y_data, 'ro', label='Original data')
    plt.plot(x_data, sess.run(W) * x_data + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()

