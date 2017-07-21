'''
CNN in TensorFlow
Weights Dimensions [Height, Width, No. Input Channels, No. Output Channels]
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Visualization
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Input data params
img_height = 28
img_width = 28
img_channels = 1
num_classes = 10

# Training params
learning_rate = 0.0001
batch_size = 100
num_epochs = 1000

# Get the data
mnist_data = input_data.read_data_sets('MNIST_data', one_hot = True)

# Model
layer1_size = 5
layer1_stride = 1
layer1_filters = 8

layer2_size = 4
layer2_stride = 2
layer2_filters = 12

layer3_size = 4
layer3_stride = 2
layer3_filters = 24

layer4_nodes = 200

# Input layer
X = tf.placeholder(tf.float32, [None, img_height * img_width])
X_img = tf.reshape(X, [-1, img_width, img_height, img_channels])
Y_ = tf.placeholder(tf.float32, [None, num_classes])

# Layer 1
W1 = tf.Variable(tf.truncated_normal([layer1_size, layer1_size, img_channels, layer1_filters], stddev = 0.1))
b1 = tf.Variable(tf.ones([layer1_filters]) / 10)
Y1 = tf.nn.relu(tf.nn.conv2d(X_img, W1, strides = [1, layer1_stride, layer1_stride, 1], padding = 'SAME') + b1)

# Layer 2
W2 = tf.Variable(tf.truncated_normal([layer2_size, layer2_size, layer1_filters, layer2_filters], stddev = 0.1))
b2 = tf.Variable(tf.ones([layer2_filters]) / 10)
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides = [1, layer1_stride, layer1_stride, 1], padding = 'SAME') + b2)

# Layer 3
W3 = tf.Variable(tf.truncated_normal([layer3_size, layer3_size, layer2_filters, layer3_filters], stddev = 0.1))
b3 = tf.Variable(tf.ones([layer3_filters]) / 10)
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides = [1, layer3_stride, layer3_stride, 1], padding = 'SAME') + b3)
# Y3_shape = Y3.get_shape()
# print Y3_shape
Y3_flat = tf.reshape(Y3, [-1, 14 * 14 * 24])#tf.reshape(Y3, [-1, (Y3_shape[1] / (layer1_stride * layer2_stride)) * (Y3_shape[2] / (layer1_stride * layer2_stride)) * Y3_shape[3]])

# Layer 4
W4 = tf.Variable(tf.truncated_normal([14 * 14 * 24, layer4_nodes], stddev = 0.1))#tf.Variable(tf.truncated_normal([Y3_shape[0] * Y3_shape[1] * Y3_shape[2], layer4_nodes], stddev = 0.1))
b4 = tf.Variable(tf.ones([layer4_nodes]) / 10)
Y4 = tf.nn.relu(tf.matmul(Y3_flat, W4) + b4)

# Layer 5
W5 = tf.Variable(tf.truncated_normal([layer4_nodes, num_classes], stddev = 0.1))
b5 = tf.Variable(tf.zeros([num_classes]))
Y = tf.nn.softmax(tf.matmul(Y4, W5) + b5)

# Loss function - Cross Entropy
cross_entropy = - tf.reduce_sum(Y_ * tf.log(Y))

# % of correct answers found in batch
is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train_step = optimizer.minimize(cross_entropy)

# Create the session
sess = tf.Session()

# Initialize the variables
sess.run(tf.global_variables_initializer())

# Visualization
x_axis = []
training_accuracy_record = []
training_loss_record = []
testing_accuracy_record = []
testing_loss_record = []


# Run our Model!!
for i in range(num_epochs):
	# Load the input data
	batch_X, batch_Y = mnist_data.train.next_batch(batch_size)
	train_data = {X: batch_X, Y_: batch_Y}

	# Train
	sess.run(train_step, feed_dict = train_data)

	# Accuracy on training data
	acc, loss = sess.run([accuracy, cross_entropy], feed_dict = train_data)

	# Trying it out on test data
	test_data = {X: mnist_data.test.images, Y_: mnist_data.test.labels}

	# Accuracy on test data
	acc_test, loss_test = sess.run([accuracy, cross_entropy], feed_dict = test_data)
	
	x_axis.append(i)
	training_accuracy_record.append(acc)
	training_loss_record.append(loss)
	testing_accuracy_record.append(acc_test)
	testing_loss_record.append(loss_test)

	# Print stuff
	print acc, loss, acc_test, loss_test

plt.plot(x_axis, training_accuracy_record, 'r-', x_axis, testing_accuracy_record, 'b-')
plt.show()
plt.plot(x_axis, training_loss_record, 'r-', x_axis, testing_loss_record, 'b-')
plt.show()
