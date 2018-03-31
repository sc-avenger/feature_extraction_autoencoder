from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers

from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm,metrics

f=open("result.txt","w+")

class autoencoder():

	def createAutoencoder(self,input):
		self.encoding_dim = 32
		self.input_img = Input(shape=(784,))
		self.encoded = Dense(self.encoding_dim, activation='relu',activity_regularizer=regularizers.l1(10e-5))(self.input_img)
		self.decoded = Dense(784, activation='sigmoid')(self.encoded)
		self.autoencoder = Model(self.input_img, self.decoded)
		encoder = Model(self.input_img, self.encoded)
		self.encoder = encoder

	def compile(self):
		self.encoder = Model(self.input_img, self.encoded)
		encoded_input = Input(shape=(self.encoding_dim,))
		decoder_layer = self.autoencoder.layers[-1]
		self.decoder = Model(encoded_input, decoder_layer(encoded_input))
		self.autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

	def train(self,trainingData,testData):
		self.autoencoder.fit(trainingData,trainingData,epochs = 50 , batch_size=256,shuffle=True,validation_data = (testData,testData))

	def test(self,data):
		return self.encoder.predict(data)

	def encodeData(self,inputData):
		return self.encoder.predict(inputData)



class SVM():

	def __init__(self,input):
		self.clf = svm.SVC()
		self.autoencoder = autoencoder()
		self.autoencoder.createAutoencoder(input)
		self.autoencoder.compile()


	def flattenImage(self,input):
		l = len(input)
		return input.reshape(l,-1)


	def train(self,trainingData,label,testData):
		#self.autoencoder.train(trainingData,testData)
		#encodedtrain = self.autoencoder.encodeData(trainingData[0:10000])
		self.encodedImage = self.flattenImage(trainingData)
		self.clf.fit(self.encodedImage,label)


	def test(self,testData):
		encodedData = testData #self.autoencoder.encodeData(testData)
		l = len(encodedData)
		encodedData  = encodedData.reshape(l,-1)
		return self.clf.predict(encodedData)




class Main():

	def __init__(self,trainingData,testData,trainLabel,testLabel):
		self.trainingData = trainingData
		self.testData = testData
		self.trainLabel = trainLabel
		self.testLabel = testLabel

	def printaccuracy(self,abel,predicted):
		c=0
		cnt = 0
		for x,y in zip(abel, predicted):
			cnt = cnt +1
			if str(x) == str(y):
				c =c + 1
		"print(c/cnt)
		f.write(str(self.trainingData)+'\n\n'+str(self.testData)+'\n\n'+str(self.trainLabel)+'\n\n'+str(self.testLabel)+'\n\n'+str(self.labels)+'\n\n'+str(c/cnt))


	def run(self):
		encoding_dim = 32
		svm=SVM(Input(shape=(encoding_dim,)))
		svm.train(self.trainingData,self.trainLabel,self.testData)
		ans = svm.test(self.testData)
		print("Classification report for classifier %s:n%sn" % (svm.clf, metrics.classification_report(self.testLabel, ans)))
		self.printaccuracy(self.testLabel,ans)







(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

main= Main(x_train,x_test,y_train,y_test)
main.run()
