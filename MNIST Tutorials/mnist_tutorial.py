#Import MNIST Dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import matplotlib.pyplot as plt
import numpy as np
import random as ran
import tensorflow as tf

#Function that creates a training set of data
def TRAIN_SIZE(num):
    x_train = mnist.train.images[:num,:]
    y_train = mnist.train.labels[:num,:]
    return x_train, y_train
   
#Function that creates a training set of data
def TEST_SIZE(num):
    x_test = mnist.test.images[:num,:]
    y_test = mnist.test.labels[:num,:]
    return x_test, y_test

#display the digit and the label
def display_train_digit(num):
    label = y_train[num].argmax(axis=0)
    image = x_train[num].reshape([28,28])
    plt.title('Label: %d' % (label))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()

#display the digit and the label (& print label to console)
def display_test_digit(num):
    ##turn the data points/arrays into strings then print them out
    label_array=str(y_test[num])
    label_list  = np.array(y_test[num]).tolist()
    print('Label Array: ' + label_array)
    print('Label: ' + str(label_list.index(max(label_list))))
    #plot the data points in a graph
    label = y_test[num].argmax(axis=0)
    image = x_test[num].reshape([28,28])
    plt.title('Label: %d' % (label))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show() 

#display the weights
def show_weight_plot():
    for i in range(10):
        plt.subplot(2, 5, i+1)
        weight = sess.run(weights)[:,i]
        plt.title(i)
        plt.imshow(weight.reshape([28,28]), cmap=plt.get_cmap('seismic'))
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)
    plt.show()

    
#Model dimensions
inputs = tf.placeholder(tf.float32, shape=[None, 784])
target = tf.placeholder(tf.float32, shape=[None, 10])
weights = tf.Variable(tf.zeros([784,10]))

#The Model:
model_output = tf.nn.sigmoid(tf.matmul(inputs,weights))


#Cost Function and Learning Algorithms
cost= tf.losses.mean_squared_error(target, model_output)

LEARNING_RATE = 0.1
TRAIN_STEPS = 2500
training = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)


#Training the model
x_train, y_train = TRAIN_SIZE(5500)
x_test, y_test = TEST_SIZE(1000)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

correct_prediction = tf.equal(tf.argmax(model_output,1), tf.argmax(target,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

for i in range(TRAIN_STEPS+1):
    sess.run(training, feed_dict={inputs: x_train, target: y_train})
    if i%100 == 0:
        print('Training Step:' + str(i) + '  Accuracy =  ' + str(sess.run(accuracy, feed_dict={inputs: x_test, target: y_test})) + '  Cost = ' + str(sess.run(cost, {inputs: x_train, target: y_train})))

#Show the trained model
show_weight_plot()


#Testing the Model
def run_trained_model(num):
    print('\nTest Number ' + str(num))
    test_num = ran.randint(0, x_test.shape[0])
    display_test_digit(test_num)    
    #runs the model on the random member of the test group, and turns it into a string 
    m_output = np.array(sess.run(model_output, {inputs: x_test[test_num].reshape(1, 784)}).reshape(10)).tolist()
    maxpos = m_output.index(max(m_output))
    print('Model Output: ' + str(m_output))
    print('Prediction: ' + str(maxpos))

for i in range(1,6):
    run_trained_model(i)
    input("Press [enter] to continue.")


#That's all folks!
print("\n\nSo long and thanks for all the fish!")


'''
    ***Improvements/Alterations***

############
# The Model (additions and changes to Lines 55-61)
############
    
#Softmaxreplace line 61
model_output = tf.nn.softmax(tf.matmul(inputs,weights))

#Softmax and Bias: replace line 61 & comment out 'show_weight_plot'
bias = tf.Variable(tf.zeros([10]))
model_output = tf.nn.softmax(tf.matmul(inputs,weights) + bias)

#Multilayer, Softmax, and Bias: replace lines 55-61
inputs = tf.placeholder(tf.float32, shape=[None, 784])
target = tf.placeholder(tf.float32, shape=[None, 10])

weights_l1 = tf.Variable(tf.truncated_normal([784,20], stddev=0.1))
bias_l1 = tf.Variable(tf.truncated_normal([20], stddev=0.1))
weights_l2 = tf.Variable(tf.truncated_normal([20,20], stddev=0.1))
bias_l2 = tf.Variable(tf.truncated_normal([20], stddev=0.1))
weights_l3 = tf.Variable(tf.truncated_normal([20,10], stddev=0.1))
bias_l3 = tf.Variable(tf.truncated_normal([10], stddev=0.1))

l1_output = tf.nn.sigmoid(tf.matmul(inputs,weights_l1)+ bias_l1)
l2_output = tf.nn.sigmoid(tf.matmul(l1_output,weights_l2) + bias_l2)
model_output = tf.nn.softmax(tf.matmul(l2_output,weights_l3) + bias_l3)

#Convolutional



################
# Cost Functions
################

#most basic
?????

#cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(target * tf.log(model_output), reduction_indices=[1]))

#####################
# Learning Algorithms
#####################

#Learning function changes
training = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)


'''