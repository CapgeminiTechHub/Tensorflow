#Import MNIST Dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

'''
    The MNIST database is very large, so to speed things up we're only going to select a sample from the full database.
    
    The below functions will define a training and test dataset.
   
    This approach also allows us to experiment with the size of training set, and how it affects training and the end model.
'''

#Function that creates a training set of data, by taking the first 'num' from the MNIST training dataset
def TRAIN_SIZE(num):
    #The code
    x_train = mnist.train.images[:num,:]
    y_train = mnist.train.labels[:num,:]
    #just writing to the console
    print ('--------------------------------------------------')
    print ('x_train Examples Loaded = ' + str(x_train.shape))
    print ('y_train Examples Loaded = ' + str(y_train.shape))
    #return the dataset
    return x_train, y_train

print('\nLoading an example training set:')
#Use the above function to create a (very small) training set
x_train, y_train = TRAIN_SIZE(5)
   
#Function that creates a training set of data, by taking the first 'num' from the MNIST training dataset 
def TEST_SIZE(num):
    #The code
    x_test = mnist.test.images[:num,:]
    y_test = mnist.test.labels[:num,:]
    #just writing to the console
    print ('--------------------------------------------------')
    print ('x_test Examples Loaded = ' + str(x_test.shape))
    print ('y_test Examples Loaded = ' + str(y_test.shape))
    #return the dataset
    return x_test, y_test



'''
    We've set up some sets of data for us to use in training and testing. Now however lets actually have a look at the data.

    Each data point(???) in this database consist of a 28x28 pixel picture of a handwritten digit (0-9), and a label denoting w. 
    
    Originally (NIST database) simply consisted of black and white pixels, they now contain greyscale due to the normalisation and anti-aliasing technique used. However even so this image can be represented as a 784 long array (list)
    
    The label consist of a 10 long array of 0's and 1, with the 1 deonting the correct digit.
    i.e. the label of '3' would be denoted by [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    
    In a neural network model, which array provides the inputs, and which array provides the target?
    
    The below functions will allow us to view a datapoint.
'''

#import some useful libraries - put this at the start of the code.
import matplotlib.pyplot as plt
import numpy as np
import random as ran

#function for printing a 784 array, in 28x28 format
def pretty_print_array(array, format="array"):
    """ This function will format an array with 784 elements into a 28 x 28 format
    By default it will present as an array (enclosed by square brackets)
    If the format is set to anything other than array it will not contain square brackets
    """
    n = 1
    pretty_string = ""

    if format == "array":
        pretty_string = pretty_string + "["

    for element in array:
        ## Every 28 elements we want to insert a new line
        if n % 28 == 0:
            ## The final element will not need a new line, but will need a closing square bracket if the format is an array
            if n == 784:
                pretty_string = pretty_string + str(element)
                if format == "array":
                    pretty_string = pretty_string + "]"
            else:
                pretty_string = pretty_string + str(element) + ",\n" 
        else:
            pretty_string = pretty_string + str(element) + ", "
        n += 1
    print(pretty_string)


#display the digit and the label
def display_train_digit(num):
    #turn the data points/arrays into strings then print them out
    #digit_array=str(x_train[num])
    digit_array=['%.2f' % elem for elem in x_train[num]]
    label_array=str(y_train[num])
    print('Label array is: ' + label_array)
    #print('Digit array is: ' + digit_array)
    pretty_print_array(digit_array)
    #plot the data points in a graph
    label = y_train[num].argmax(axis=0)
    image = x_train[num].reshape([28,28])
    plt.title('Label: %d' % (label))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()
    
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

print('\nDisplaying a random digit from the training dataset:')    
#This will display a random datapoint from the training dataset
display_train_digit(ran.randint(0, x_train.shape[0]-1))

'''
    The Model
    
    We haven't actually used Tensorflow yet. We've grabbed the MNIST database, and done a visualisation of it, but  we haven't touched on the predictive model yet.
        
'''

#Import tensorflow - put this at the start of the code.
import tensorflow as tf

#Placeholders are variables that we will assign data to - generally used as the inputs and outputs
inputs = tf.placeholder(tf.float32, shape=[None, 784])
target = tf.placeholder(tf.float32, shape=[None, 10])
#The None means we can feed it as many examples as we want.

#Variables are the trainable dimensions of the model - the weights and the threshold.
weights = tf.Variable(tf.zeros([784,10]))
#bias = tf.Variable(tf.zeros([10]))
#We currently initialise all these learning variables as Zero - we could also initialise them as a random number to create different starting conditions.

#The Model:
model_output = tf.nn.sigmoid(tf.matmul(inputs,weights))
#model_output = tf.nn.sigmoid(tf.matmul(inputs,weights) + bias)
#model = tf.nn.softmax(tf.matmul(inputs,weights) + bias)


'''
    Running the Model
    
    The above merely defines the model. We now need to initialise and run the model to actually get an output. 
    
'''

#Create a tensorflow session
sess = tf.Session()
#Run the Tensorflow session so the variable are initialised
sess.run(tf.global_variables_initializer())
#If using TensorFlow prior to 0.12 use:
#sess.run(tf.initialize_all_variables())

print('\nPrinting the initialised Model Output:')
#Now you can run the model and print the output
model_first_run=sess.run(model_output, feed_dict={inputs: x_train})
print(model_first_run)


'''
    Visualising the Model
    
    We can also write some code that helps us open the black box. The below code will map out the weights for each pixel of the 28x28 image.
'''
#Now the model has actually been run, we can also create a visualisation of the weights.
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

show_weight_plot()

'''
    Error/'Cost Function'
    
    Let start moving towards helping this model learn. For this example the model will undergo supervised learning. 
    
    In order to learn the model needs to know if it is wrong, and how wrong it is.

'''

#Complicated  equation incoming - Mean Squared Error cost function
cost= tf.losses.mean_squared_error(target, model_output)


'''
    Learning Algorithms

    The learning algorithm itself is a bit beyond this tutorial.
    
    If you wish to do some research yourself, looking into Gradient Descent and Backpropogation is a good place to start.
'''
print('\nReloading the datasets:')
#Create a train dataset - we did this above in the visualisations but now we want a larger dataset for training.
x_train, y_train = TRAIN_SIZE(5500)
#Create a test dataset
x_test, y_test = TEST_SIZE(1000)

#Set the Learning Rate - how quickly the model should learn
LEARNING_RATE = 0.1
#What are potential issues with the model learning 'too quickly'

#How many times we train using ALL training examples (ie. How many 'Epochs')
TRAIN_STEPS = 2500

#define the actual training - this is a predefined learning function.
training = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)
#how does the model know to train the above 'weights' & 'bias' variables?


'''
    The training of the model
    
    Like above with the model itself, we defined the model, but then we needed to run it. We have defined the training algorithms, but now we need the run them on the model.
'''
#Another way to initialised variables - why do we need this here again?
init = tf.global_variables_initializer()
sess.run(init)

#some commands that will be used below for console output.
correct_prediction = tf.equal(tf.argmax(model_output,1), tf.argmax(target,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print('\nTraining the Model:')
#A loop that runs the training defined above, for as many Epochs as set in TRAIN_STEPS 
for i in range(TRAIN_STEPS+1):
    #Command to run the training
    sess.run(training, feed_dict={inputs: x_train, target: y_train})
    #feed_dict - feeds the training data
    
    #The below simply outputs the Accuracy and Cost of every 100th Epoch, so we can see how the training progresses; doesn't affect the training itself.
    if i%100 == 0:
        print('Training Step:' + str(i) + '  Accuracy =  ' + str(sess.run(accuracy, feed_dict={inputs: x_test, target: y_test})) + '  Cost = ' + str(sess.run(cost, {inputs: x_train, target: y_train})))

#update the weight visualisation - lets see what all the weights look like now
show_weight_plot()


'''
Testing the Model

Normally testing would be a rigorous unbiased procedure to ascertain whether the model has successfully learnt and generalised.

However as this is simply a tutorial our testing will merely consist of us randomly picking 5 examples, and seeing if the model gets the correct prediction.
'''
#Testing the Model
def run_trained_model(num):
    test_num = ran.randint(0, x_test.shape[0])
    display_test_digit(test_num)    
    #runs the model on the random member of the test group, and turns it into a string 
    m_output = np.array(sess.run(model_output, {inputs: x_test[test_num].reshape(1, 784)}).reshape(10)).tolist()
    maxpos = m_output.index(max(m_output))
    print('Model Output: ' + str(m_output))
    print('Prediction: ' + str(maxpos))

for i in range(0,5):
    alphabet = ['A','B','C','D','E','F']
    print('\nTest ' + alphabet[i])
    run_trained_model(i)
    input("Press [enter] to continue.")


#That's all folks!
print("\n\nSo long and thanks for all the fish!")
