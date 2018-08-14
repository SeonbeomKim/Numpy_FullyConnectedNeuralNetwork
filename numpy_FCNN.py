import FCNN_class as fc
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data #for MNIST dataset

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_set = np.hstack((mnist.train.images, mnist.train.labels)) #shape = 55000, 794   => 784개는 입력, 10개는 정답.
vali_set = np.hstack((mnist.validation.images, mnist.validation.labels))
test_set = np.hstack((mnist.test.images, mnist.test.labels))

lr = 0.01

def train(model, data):
	batch_size = 64
	loss = 0
	np.random.shuffle(data)

	for i in range( int(np.ceil(len(data)/batch_size)) ):
		#print(i+1,'/',int(np.ceil(len(data)/batch_size)))
		batch = data[batch_size * i: batch_size * (i + 1)]
		input_ = batch[:, :784]
		target_ = batch[:, 784:]
	
		logits = model.forward(input_)
		train_loss = model.backward(logits, target_, lr=lr)
		loss += train_loss
	
	return loss/len(data)


def validation(model, data):
	batch_size = 64
	loss = 0
	
	for i in range( int(np.ceil(len(data)/batch_size)) ):
		batch = data[batch_size * i: batch_size * (i + 1)]
		input_ = batch[:, :784]
		target_ = batch[:, 784:]
	
		logits = model.forward(input_)
		vali_loss = model.calc_loss(logits, target_)
		loss += vali_loss
	
	return loss/len(data)


def test(model, data):
	batch_size = 64
	correct = 0

	for i in range( int(np.ceil(len(data)/batch_size)) ):
		batch = data[batch_size * i: batch_size * (i + 1)]
		input_ = batch[:, :784]
		target_ = batch[:, 784:]

		logits = model.forward(input_)
		check = model.correct(logits, target_, axis=1)
		correct += check

	return correct/len(data)


def run(model, train_set, vali_set, test_set):
	for epoch in range(1, 300):
		train_loss = train(model, train_set)
		vali_loss = validation(model, vali_set)
		accuracy = test(model, test_set)

		print("epoch:", epoch, "\ttrain_loss:", train_loss, "\tvali_loss:", vali_loss, "\taccuracy:", accuracy)


model = fc.model()
model.connect(fc.affine(w_shape=[784, 128], b_shape=[128]))
model.connect(fc.relu())
model.connect(fc.affine(w_shape=[128, 128], b_shape=[128]))
model.connect(fc.relu())
model.connect(fc.affine(w_shape=[128, 10], b_shape=[10]))
model.connect_loss(fc.softmax_cross_entropy_with_logits())

run(model, train_set, vali_set, test_set)



'''
#xor
x = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [[1, 0], [0, 1], [0, 1], [1, 0]]


model = fc.model()
model.connect(fc.affine(w_shape=[2, 10], b_shape=[10]))
model.connect(fc.relu())
model.connect(fc.affine(w_shape=[10, 10], b_shape=[10]))
model.connect(fc.relu())
model.connect(fc.affine(w_shape=[10, 2], b_shape=[2]))
model.connect_loss(fc.softmax_cross_entropy_with_logits())


for i in range(300):
	logits = model.forward(x)
	loss = model.backward(logits, y, lr=0.1)
	print(logits, loss)

print(model.correct(logits, y, axis=1))

'''