from gensim import models as g_models
from scipy.spatial.distance import cosine
import numpy as np
from keras.layers import Input
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.normalization import *
from keras.optimizers import *
from keras.models import Model
from tqdm import tqdm
import io
import os
import nltk

os.environ["KERAS_BACKEND"] = "theano"


max_sentence_length = 5

raw_text_file = open('../../data/babi_data.txt')
model = g_models.Word2Vec.load('../../models/babi_model')

hidden_neurons = 300


def get_vector_for_sentence(sentence):
    decoded_sentence = nltk.sent_tokenize(sentence.lower().decode('utf-8'))[0]
    tokens = nltk.word_tokenize(decoded_sentence)
    token_vectors = []
    for token in tokens:
        token_vectors.append(model[token])
    token_vectors = np.array(token_vectors)
    return token_vectors.reshape(token_vectors.shape[0] * token_vectors.shape[1])


raw_text_vectors = []
for f in raw_text_file:
    vector = get_vector_for_sentence(f)
    raw_text_vectors.append(vector)
raw_text_vectors = np.array(raw_text_vectors)
vector_size = raw_text_vectors.shape[1] / max_sentence_length


def get_sentence_from_vector(vector):
    v_tokens = [vector[i:i + vector_size] for i in range(0, len(vector), vector_size)]
    words = []
    for v in v_tokens:
        vocab = [word for word in model.vocab]
        words.append(min(vocab, key=lambda x: cosine(model[x], v)))
    return words


opt = Adam(lr=1e-3)
dopt = Adam(lr=1e-2)
size_of_sent_vector = max_sentence_length * vector_size

# Build Generative model ...
generator_model_input = Input(shape=[size_of_sent_vector])
generator_hidden_layer = Dense(hidden_neurons, init='glorot_normal')(generator_model_input)
generator_hidden_layer = Dropout(0.1)(generator_hidden_layer)
generator_hidden_layer = BatchNormalization(mode=2)(generator_hidden_layer)
generator_hidden_layer = Activation('relu')(generator_hidden_layer)
generator_hidden_layer = Dense(hidden_neurons, init='glorot_normal')(generator_hidden_layer)
generator_hidden_layer = Dropout(0.1)(generator_hidden_layer)
generator_hidden_layer = BatchNormalization(mode=2)(generator_hidden_layer)
generator_hidden_layer = Activation('relu')(generator_hidden_layer)
generator_hidden_layer = Dense(hidden_neurons, init='glorot_normal')(generator_hidden_layer)
generator_hidden_layer = Dropout(0.1)(generator_hidden_layer)
generator_hidden_layer = BatchNormalization(mode=2)(generator_hidden_layer)
generator_hidden_layer = Activation('relu')(generator_hidden_layer)
generator_hidden_layer = Dense(hidden_neurons, init='glorot_normal')(generator_hidden_layer)
generator_hidden_layer = Dropout(0.1)(generator_hidden_layer)
generator_hidden_layer = BatchNormalization(mode=2)(generator_hidden_layer)
generator_hidden_layer = Activation('relu')(generator_hidden_layer)
generator_output = Dense(size_of_sent_vector, activation='relu')(generator_hidden_layer)
generator = Model(generator_model_input, generator_output)
generator.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])


# Build Discriminative model ...
discriminator_model_input = Input(shape=[size_of_sent_vector])
discriminator_hidden_layer = Dense(hidden_neurons, init='glorot_normal')(discriminator_model_input)
discriminator_hidden_layer = Dropout(0.5)(discriminator_hidden_layer)
discriminator_hidden_layer = Activation('relu')(discriminator_hidden_layer)
discriminator_hidden_layer = Dense(hidden_neurons, init='glorot_normal')(discriminator_hidden_layer)
discriminator_hidden_layer = Dropout(0.5)(discriminator_hidden_layer)
discriminator_hidden_layer = Activation('relu')(discriminator_hidden_layer)
discriminator_hidden_layer = Dense(hidden_neurons, init='glorot_normal')(discriminator_hidden_layer)
discriminator_hidden_layer = Dropout(0.5)(discriminator_hidden_layer)
discriminator_hidden_layer = Activation('relu')(discriminator_hidden_layer)
discriminator_hidden_layer = Dense(hidden_neurons, init='glorot_normal')(discriminator_hidden_layer)
discriminator_hidden_layer = Dropout(0.5)(discriminator_hidden_layer)
discriminator_hidden_layer = Activation('relu')(discriminator_hidden_layer)
discriminator_hidden_layer = Dense(hidden_neurons, init='glorot_normal')(discriminator_hidden_layer)
discriminator_hidden_layer = Dropout(0.5)(discriminator_hidden_layer)
discriminator_hidden_layer = Activation('relu')(discriminator_hidden_layer)
discriminator_output = Dense(2, activation='softmax')(discriminator_hidden_layer)
discriminator = Model(discriminator_model_input, discriminator_output)
discriminator.compile(loss='categorical_crossentropy', optimizer=dopt, metrics=["accuracy"])

# Build stacked GAN model
gan_input = Input(shape=[size_of_sent_vector])
gan_hidden_layer = generator(gan_input)
gan_V = discriminator(gan_hidden_layer)
GAN = Model(gan_input, gan_V)
GAN.compile(loss='categorical_crossentropy', optimizer=opt)
losses = []


def set_training_status(model_structure, status):
    model_structure.trainable = status
    for layer in model_structure.layers:
        layer.trainable = status


def train_for_n(nb_epoch=5000, BATCH_SIZE=50):

    for e in tqdm(range(nb_epoch)):

        batch_from_raw_text = raw_text_vectors[np.random.randint(0, len(raw_text_vectors), size=BATCH_SIZE)]
        # noise_for_generator = np.random.normal(4, 1, size=[BATCH_SIZE, size_of_sent_vector])
        noise_for_generator = np.random.uniform(0, 1, size=[BATCH_SIZE, size_of_sent_vector])
        generated_text_from_noise = generator.predict(noise_for_generator)

        discriminator_input = np.concatenate((batch_from_raw_text, generated_text_from_noise))
        discriminator_output_labels = np.zeros([2 * BATCH_SIZE, 2])
        discriminator_output_labels[0:BATCH_SIZE, 1] = 1
        discriminator_output_labels[BATCH_SIZE:, 0] = 1

        discriminator_loss = discriminator.train_on_batch(discriminator_input, discriminator_output_labels)
        losses.append(discriminator_loss[0])

        noise_for_gan = np.random.uniform(0, 1, size=[BATCH_SIZE, size_of_sent_vector])
        # noise_for_gan = np.random.normal(4, 1, size=[BATCH_SIZE, size_of_sent_vector])
        labels_for_gan = np.zeros([BATCH_SIZE, 2])
        labels_for_gan[:, 1] = 1

        generator_loss = GAN.train_on_batch(noise_for_gan, labels_for_gan)

        if e % 100 == 0:
            print 'Discriminator Accuracy', discriminator_loss[0]
            print 'Generator Accuracy', generator_loss
            for sentence in generated_text_from_noise:
                print ' '.join(get_sentence_from_vector(sentence))


loss_data = open('d_loss_scores.txt', 'w')
for loss in losses:
    loss_data.write(str(loss) + '\n')
loss_data.close()

def generate_sentences(number_of_sentences):
    set_training_status(generator, False)
    output_file = io.open('GAN_20_Words.txt', 'w', encoding='utf-8')
    noise_gen = np.random.uniform(0, 1, size=[number_of_sentences, size_of_sent_vector])
    sentences = generator.predict(noise_gen)
    for sentence in sentences:
        sentence_words = ' '.join(get_sentence_from_vector(sentence))
        output_file.write(sentence_words + '\n')
    print 'Sentences written to file'


opt.lr.set_value(1e-4)
# opt.beta_1 = 0.5
dopt.lr.set_value(1e-4)
# dopt.beta_1 = 0.5
train_for_n(nb_epoch=3000, BATCH_SIZE=10)

generate_sentences(150)
