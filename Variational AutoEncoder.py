
import os,random
os.environ["KERAS_BACKEND"] = "tensorflow"
from gensim import models as g_models
from scipy.spatial.distance import cosine
#os.environ["THEANO_FLAGS"]  = "device=gpu%d,lib.cnmem=0"%(random.randint(0,3))
import numpy as np
import theano as th
import theano.tensor as T
from keras.utils import np_utils
import keras.models as models
from keras.layers import Input,merge
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten,MaxoutDense
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import *
from keras.layers.wrappers import TimeDistributed
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D
from keras.layers.recurrent import LSTM
from keras.regularizers import *
from keras.layers.normalization import *
from keras.optimizers import *
from keras.datasets import mnist
import matplotlib.pyplot as plt
#import seaborn as sns
import cPickle, random, sys, keras
from keras.models import Model
#from IPython import display
from keras.utils import np_utils
from tqdm import tqdm
model = g_models.Word2Vec.load('model')

file = open('../../data/babi/babi_data.txt')
def createVector(s):
	tokens  = s.split()
	return np.concatenate([model[t] for t in tokens])
X_train = []
for f in file:
	X_train.append(createVector(f))
X_train = np.array(X_train)
def clamp(t):
	v_tokens = [t[i:i+100] for i in range(0, len(t), 100)]
	tokens = []
	for v in v_tokens:
		vocab = [word for word in model.vocab]
		tokens.append( min(vocab, key=lambda x: cosine(model[x],v)))
	print tokens


opt = Adam(lr=1e-4)
dopt = Adam(lr=1e-3)
size_of_sent_vector = 5*100
# Build Generative model ...
g_input = Input(shape=[100])
H = Dense(100, init='glorot_normal')(g_input)
H = BatchNormalization(mode=2)(H)
H = Activation('relu')(H)
H = Dense(100, init='glorot_normal')(H)
H = BatchNormalization(mode=2)(H)
H = Activation('relu')(H)
H = Dense(size_of_sent_vector, init='glorot_normal')(H)
H = BatchNormalization(mode=2)(H)

g_V = Activation('linear')(H)
generator = Model(g_input,g_V)
generator.compile(loss='binary_crossentropy', optimizer=opt)
generator.summary()


# Build Discriminative model ...
d_input = Input(shape=[size_of_sent_vector])
H = Dense(100, init='glorot_normal')(d_input)
H = BatchNormalization(mode=2)(H)
H = Activation('relu')(H)
H = Dense(1000, init='glorot_normal')(H)
H = BatchNormalization(mode=2)(H)
H = Activation('relu')(H)
d_V = Dense(2,activation='sigmoid')(H)
discriminator = Model(d_input,d_V)
discriminator.compile(loss='categorical_crossentropy', optimizer=dopt)
discriminator.summary()


# Freeze weights in the discriminator for stacked training
def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val
#make_trainable(discriminator, False)

# Build stacked GAN model
gan_input = Input(shape=[100])
H = generator(gan_input)
gan_V = discriminator(H)
GAN = Model(gan_input, gan_V)
GAN.compile(loss='categorical_crossentropy', optimizer=opt)
GAN.summary()
losses = {"d":[], "g":[]}

def train_for_n(nb_epoch=5000, plt_frq=25,BATCH_SIZE=32):

    for e in tqdm(range(nb_epoch)):  
        
        # Make generative images

        text_batch = X_train[np.random.randint(0,len(X_train),size=BATCH_SIZE)]    
        noise_gen = np.random.uniform(0,1,size=[BATCH_SIZE,100])
        generated_text = generator.predict(noise_gen)
        if e%500 == 0:
	        for generated in generated_text:
	        	clamp(generated)
        
        # Train discriminator on generated images
        X = np.concatenate((text_batch, generated_text))
        y = np.zeros([2*BATCH_SIZE,2])
        y[0:BATCH_SIZE,1] = 1
        y[BATCH_SIZE:,0] = 1
        
        #make_trainable(discriminator,True)
        d_loss  = discriminator.train_on_batch(X,y)
        if e%100 ==0:
        	print d_loss
        losses["d"].append(d_loss)
    
        # train Generator-Discriminator stack on input noise to non-generated output class
        noise_tr = np.random.uniform(0,1,size=[BATCH_SIZE,100])
        y2 = np.zeros([BATCH_SIZE,2])
        y2[:,1] = 1
        
        #make_trainable(discriminator,False)
        g_loss = GAN.train_on_batch(noise_tr, y2 )
        losses["g"].append(g_loss)
        

train_for_n(nb_epoch=1000, plt_frq=500,BATCH_SIZE=32)

# Train for 2000 epochs at reduced learning rates
opt.lr.set_value(1e-5)
dopt.lr.set_value(1e-4)
train_for_n(nb_epoch=1000, plt_frq=500,BATCH_SIZE=32)

# Train for 2000 epochs at reduced learning rates
opt.lr.set_value(1e-6)
dopt.lr.set_value(1e-5)
train_for_n(nb_epoch=1000, plt_frq=500,BATCH_SIZE=32)
