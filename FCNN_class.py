import numpy as np

class affine:
	def __init__(self, w_shape, b_shape):
		self.x = None # input
		self.w = np.random.randn(*w_shape)*0.1 # weight, list자체로 받지 못해서 *로 전달.
		self.b = np.zeros(b_shape) # bias
		self.dw = None # w gradient
		self.db = None # bias gradient

	def forward(self, x):
		self.x = x # input
		return np.matmul(x, self.w) + self.b

	def backward(self, grad=1):
		#x.T = [w_shape[0], batch], grad = [batch, w_shape[1]]  즉 np.matmul(x.T, grad) 하면 batch전체에 관해 dw가 계산됨.
		self.dw = np.matmul(self.x.T, grad) #shape 때문에 이렇게 됨. 계산그래프 그려보면 이해됨.
		self.db = np.mean(grad, axis=0) #batch별 평균.
		return np.matmul(grad, self.w.T) # x에 관해서 계속 backpropagation 되기 때문에 x에관한 미분을 리턴해서 이전 layer에 전파.

class relu:
	def __init__(self):
		self.mask = None

	def forward(self, x):
		mask = (x<=0) # 0이하인 값 mask,  if x = [10, -1, 3] => mask = [false, true, false]
		self.mask = mask # backward할 때, mask된 부분(0이하인 값)은 미분 0
		x[mask] = 0 # 0이하인 값에 0 할당.
		return x

	def backward(self, grad):
		grad[self.mask] = 0 #forward 값이 0이하였던 부분은 미분값 0으로 할당.
		return grad

class sigmoid:
	def __init__(self):
		self.sigvalue = None 

	def forward(self, x):
		sigvalue = 1/(1+np.exp(-x))
		self.sigvalue = sigvalue
		return sigvalue

	def backward(self, grad=1):
		return grad*self.sigvalue*(1-self.sigvalue)

class softmax_cross_entropy_with_logits:
	def __init__(self):
		self.target = None
		self.pred = None #softmax 결과
		self.loss = None 

	def forward(self, x, target):
		target = np.array(target)
		self.target = target

		#softmax
		max_value = np.max(x, axis=1, keepdims=True)
		exp = np.exp(x - max_value) #max값을 빼도, 빼지 않은 것과 결과는 동일하며, 빼지 않으면 값 overflow 발생 가능. 
		pred = exp / np.sum(exp, axis=1, keepdims=True)
		self.pred = pred

		#cross_entropy
		epsilon = 1e-07
		loss = -target*np.log(pred + epsilon) # pred가 0이면 np.log = -inf
		loss = np.mean(np.sum(loss, axis=1), axis=0) #data별로 sum 하고, batch별로 mean
		self.loss = loss
		return loss

	def backward(self, grad=1):
		#return np.mean(self.pred-self.target, axis=0) #배치별로 gradient 평균냄.
		return (self.pred-self.target)/self.target.shape[0] #배치사이즈로 나눠줌. 여기서 안나누고 affine.backward에서 나눠도되긴함
		#근데 affine.backward에서 나누면 affine 레이어마다 나눠야해서 계산량이 더 많음.


class model:
	def __init__(self):
		self.graph = []
		self.loss_graph = None
		#self.output_layer = None

	def connect(self, node):
		self.graph.append(node)

	def connect_loss(self, node):
		self.loss_graph = node

	def forward(self, x):
		x = np.array(x)
		
		logits = x
		for node in self.graph:
			logits = node.forward(logits)
		
		#self.output_layer = logits
		return logits

	def backward(self, logits, y, lr=0.002): #train
		y = np.array(y)

		#calc loss
		loss = self.calc_loss(logits, y)
		#loss = self.loss_graph.forward(logits, y)
		
		#backpropagation
		grad = self.loss_graph.backward()
		for index in range(len(self.graph)-1, -1, -1):
	 		grad = self.graph[index].backward(grad) #grad 계산하면 dw,db 갱신됨.
	 		#계산그래프여서 backpropagation에 필요한건 grad에 계산되어있음. 그러므로 grad 구하자마자 w, b 업데이트해도됨. 
	 		#계산그래프 안쓰면 모든 w', b' 계산이 끝난 후 update해야함.
	 		if 'affine' in str(self.graph[index]):
	 			self.graph[index].w -= lr * self.graph[index].dw
	 			self.graph[index].b -= lr * self.graph[index].db
		return loss

	def calc_loss(self, logits, y):
		loss = self.loss_graph.forward(logits, y)
		return loss

	def correct(self, logits, y, axis=1):
		compare = (np.argmax(logits, axis) == np.argmax(y, axis))
		return np.sum(compare)


'''
class mul:
	def __init__(self):
		self.a = None
		self.b = None

	def forward(self, a, b):
		self.a = a
		self.b = b
		return a*b

	def backward(self, grad=1):
		return grad*self.b, grad*self.a

class add:
	def forward(self, a, b):
		return a+b

	def backward(self, grad=1):
		return grad, grad
'''