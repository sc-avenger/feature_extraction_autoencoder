from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers

from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm,metrics

f=open("result1.txt","w+")

class autoencoder():

	def createAutoencoder(self,input,enc_dim):
		self.encoding_dim = enc_dim
		self.input_img = Input(shape=(784,))
		self.encoded = Dense(self.encoding_dim, activation='relu')(self.input_img)
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

	def __init__(self,input,enc_dim,labels):
		self.clf = svm.SVC()
		self.autoencoder = autoencoder()
		self.autoencoder.createAutoencoder(input,enc_dim)
		self.autoencoder.compile()
		self.labels = labels


	def flattenImage(self,input):
		l = len(input)
		self.encodedImages = input.reshape(l,-1)


	def train(self,trainingData,label,testData):
		self.autoencoder.train(trainingData,testData)
		encodedtrain = self.autoencoder.encodeData(trainingData[0:self.labels])

		self.flattenImage(encodedtrain)
		ans={}

		self.clf.fit(self.encodedImages,label[0:self.labels])


	def test(self,testData):
		encodedData = self.autoencoder.encodeData(testData)
		l = len(encodedData)
		#print(encodedData.shape)
		encodedData  = encodedData.reshape(l,-1)
		return self.clf.predict(encodedData)




class Main():

	def __init__(self,trainingData,testData,trainLabel,testLabel,enc_dim,labels):
		self.trainingData = trainingData
		self.testData = testData
		self.trainLabel = trainLabel
		self.testLabel = testLabel
		self.enc_dim = enc_dim
		self.labels = labels

	def printaccuracy(self,abel,predicted):
		c=0
		oc=0
		for x,y in zip(abel, predicted):
			if(str(x) == str(y)):
				c=c+1
			oc = oc + 1
		f.write(str(self.enc_dim)+'\n\n'+str(self.trainingData)+'\n\n'+str(self.testData)+'\n\n'+str(self.trainLabel)+'\n\n'+str(self.testLabel)+'\n\n'+str(self.labels)+'\n\n'+str(c/oc))


	def run(self):
		encoding_dim = self.enc_dim
		svm=SVM(Input(shape=(encoding_dim,)),self.enc_dim,self.labels)
		svm.train(self.trainingData,self.trainLabel,self.testData)
		ans = svm.test(self.testData)
		#print("Classification report for classifier %s:n%sn" % (svm.clf, metrics.classification_report(self.testLabel, ans)))
		self.printaccuracy(self.testLabel,ans)





(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


encodingDimension = [256]
labelLength = [200]
for x in encodingDimension:
	for y  in labelLength:
		main= Main(x_train,x_test,y_train,y_test,x,y)
		main.run()

f.close()
