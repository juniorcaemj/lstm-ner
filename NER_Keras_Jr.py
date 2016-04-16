# -*- coding: utf-8 -*-
import gzip

from keras.layers.core import Dropout, TimeDistributedDense, Dense, Flatten
from keras.layers.recurrent import *
from keras.models import *
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils

import PortEvalReader
from KerasLayer.FixedEmbedding import FixedEmbedding

windowSize = 3 # 3 to the left, 3 to the right
number_of_epochs = 50
#minibatch_size = 5000
minibatch_size = 20
LEARNING_RATE = 0.1

trainFile = 'data/corpus_paramopama+second_harem.txt'
testFile = 'data/test_set_1.txt'
testFile = 'data/corpus_miniHAREM_semTempo.txt'
#testFile = 'data/corpus_miniHAREM.txt'

#####################
#
# Read in the vocab
#
#####################
print "Read in the vocab"
vocabPath =  'embeddings/Portuguese.vocab.gz'

word2Idx = {} #Maps a word to the index in the embeddings matrix
embeddings = [] #Embeddings matrix

with gzip.open(vocabPath, 'r') as fIn:
    idx = 0
    for line in fIn:
        split = line.strip().split(' ')
        embeddings.append(np.array([float(num) for num in split[1:]]))
        word2Idx[split[0]] = idx
        idx += 1

embeddings = np.asarray(embeddings, dtype='float32')

# Create a mapping for our labels
label2Idx = {'O':0}
idx = 1

# Adding remaining labels
for nerClass in ['PESSOA', 'LOCAL', 'ORGANIZACAO', 'TEMPO']:
    label2Idx[nerClass] = idx
    idx += 1

#Inverse label mapping
idx2Label = {v: k for k, v in label2Idx.items()}

#Number of neurons
n_in = 2*windowSize+1
n_hidden = n_in*embeddings.shape[1]
n_out = len(label2Idx)
MAX_LENGTH = 183


# Read in data
print "Read in data and create matrices"
#train_dev = PortEvalReader.readFile(trainFile)
#train_sentences = train_dev[:(int)(len(train_dev)*0.9)]
train_sentences = PortEvalReader.readFile(trainFile)
#dev_sentences = train_dev[(int)(len(train_dev)*0.9):]
dev_sentences = PortEvalReader.readFile(testFile)
#test_sentences = PortEvalReader.readFile(testFile)


# Create numpy arrays
train_x, train_y = PortEvalReader.createNumpyArray(train_sentences, windowSize, word2Idx, label2Idx)
dev_x, dev_y = PortEvalReader.createNumpyArray(dev_sentences, windowSize, word2Idx, label2Idx)

#train_x, train_y = PortEvalReader.createNumpyArrayWithTime(train_sentences, windowSize, word2Idx, label2Idx, embeddings)
#dev_x, dev_y = PortEvalReader.createNumpyArrayWithTime(dev_sentences, windowSize, word2Idx, label2Idx, embeddings)

#train_x, train_y = PortEvalReader.createNumpyArrayLSTM(train_sentences, word2Idx, label2Idx, embeddings)
#dev_x, dev_y = PortEvalReader.createNumpyArrayLSTM(dev_sentences, word2Idx, label2Idx, embeddings)

#Pad Sequences
#train_x = pad_sequences(train_x,value=1.)
#train_y = pad_sequences(train_y)
#dev_x  = pad_sequences(dev_x,value=1.)
#dev_y =  pad_sequences(dev_y)

#Create one-hot entity vector, e.g. [1,0,0,0,0]
#train_y = np.equal.outer(train_y, np.arange(5)).astype(np.int32)
#dev_y = np.equal.outer(dev_y, np.arange(5)).astype(np.int32)

#####################################
#
# Create the  Network
#
#####################################

print "Embeddings shape",embeddings.shape

model = Sequential()
# Embeddings layers, lookups the word indices and maps them to their dense vectors. FixedEmbeddings are _not_ updated during training
# If you switch it to an Embedding-Layer, they will be updated (training time increases significant)   
'''model.add(FixedEmbedding(output_dim=embeddings.shape[1], input_dim=embeddings.shape[0], input_length=n_in,  weights=[embeddings]))

# Flatten concatenates the output of the EmbeddingsLayer. EmbeddingsLayer gives us a 5x100 dimension output, Flatten converts it to 500 dim. vector
#model.add(Flatten())

# Hidden + Softmax Layer
model.add(LSTM(output_dim=n_hidden, init='glorot_uniform', activation='tanh', batch_input_shape=(None,n_in,embeddings.shape[1]),return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(output_dim=n_hidden, init='glorot_uniform', activation='tanh',batch_input_shape=(None,n_in,n_hidden)))
model.add(Dense(output_dim=n_out, activation='softmax'))'''

'''
model.add(FixedEmbedding(output_dim=embeddings.shape[1], input_dim=embeddings.shape[0], input_length=MAX_LENGTH,  weights=[embeddings]))

# Flatten concatenates the output of the EmbeddingsLayer. EmbeddingsLayer gives us a 5x100 dimension output, Flatten converts it to 500 dim. vector
#model.add(Flatten())

# Hidden + Softmax Layer
model.add(LSTM(output_dim=MAX_LENGTH, init='glorot_uniform', activation='tanh',batch_input_shape=(None,MAX_LENGTH, embeddings.shape[1]),return_sequences=True,))
model.add(Dropout(0.5))
model.add(LSTM(output_dim=MAX_LENGTH, init='glorot_uniform', activation='tanh',batch_input_shape=(None,MAX_LENGTH, embeddings.shape[1]),return_sequences=True,))
model.add(TimeDistributedDense(output_dim=n_out, activation='softmax'))
'''

# Embeddings layers, lookups the word indices and maps them to their dense vectors. FixedEmbeddings are _not_ updated during training
# If you switch it to an Embedding-Layer, they will be updated (training time increases significant)
model.add(FixedEmbedding(output_dim=embeddings.shape[1], input_dim=embeddings.shape[0], input_length=n_in,  weights=[embeddings]))

# Flatten concatenates the output of the EmbeddingsLayer. EmbeddingsLayer gives us a 5x100 dimension output, Flatten converts it to 500 dim. vector
model.add(Flatten())

# Hidden + Softmax Layer
model.add(Dense(output_dim=n_hidden, init='glorot_uniform', activation='tanh',))
model.add(Dense(output_dim=n_hidden, init='glorot_uniform', activation='tanh',))
model.add(Dense(output_dim=n_out, activation='softmax'))

# Use as training function SGD or Adam
model.compile(loss='categorical_crossentropy', optimizer='adagrad')

# Plotting network
#from keras.utils.visualize_util import plot
#plot(model, to_file='model.png')
#exit(0)

#sgd = SGD(lr=0.1, decay=1e-6, momentum=0.0, nesterov=False)
#model.compile(loss='categorical_crossentropy', optimizer=sgd)


print train_x.shape[0], ' train samples'
print train_x.shape[1], ' train dimension'
#print test_x.shape[0], ' test samples'

# Train_y is a 1-dimensional vector containing the index of the label
# With np_utils.to_categorical we map it to a 1 hot matrix
train_y_cat = np_utils.to_categorical(train_y, n_out)
#dev_y_cat = np_utils.to_categorical(dev_y, n_out)

##################################
#
# Training of the Network
#
##################################

print "%d epochs" % number_of_epochs
print "%d mini batches" % (len(train_x)/minibatch_size)

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

model.load_weights('modelos_keras/modelo_keras_MLP.h5')

### Testing ###
tp = 0.0
fp = 0.0
fn = 0.0

for batch in iterate_minibatches(dev_x, dev_y, 2000, shuffle=False):
    inputs, a = batch
    b = model.predict_classes(inputs, verbose=0)
    label_y = [idx2Label[element] for element in a]
    pred_labels = [idx2Label[element] for element in b]
    for i in xrange(0,len(label_y)):
        if pred_labels[i] <> 'O' and label_y[i] <> 'O' and pred_labels[i] == label_y[i]:
            tp += 1
        elif pred_labels[i] == 'O' and label_y[i] <> 'O':
            fn += 1
        elif pred_labels[i] <> 'O' and label_y[i] <> pred_labels[i]:
            fp += 1
try:
    prec = tp/(tp+fp)
    rec = tp/(tp+fn)
    fmes = 2*(prec*rec)/(prec+rec)
    print ("True Positives: {:.1f}".format(tp))
    print ("False Positives: {:.1f}".format(fp))
    print ("False Negatives: {:.1f}".format(fn))
    print ("Precision: {:.6f}".format(prec))
    print ("Recall: {:.6f}".format(rec))
    print ("F-Measure: {:.6f}".format(fmes))
    print ("-----------------------------------------------\n")
except:
    print ("Erro de divisão por zero. Continuando...")
exit(0)
'''
model.load_weights('modelos_keras/modelo_keras_LSTM_semjanela.h5')

### Testing ###
tp = 0.0
fp = 0.0
fn = 0.0

for batch in iterate_minibatches(dev_x, dev_y, 100, shuffle=False):
    inputs, a = batch
    b = model.predict_classes(inputs, verbose=0)
    label_y = []
    pred_labels = []
    for sentence in a:
        for element in sentence:
            label_y.append(idx2Label[np.argmax(element)])
    for sentence in b:
        for element in sentence:
            pred_labels.append(idx2Label[element])
    for i in xrange(0,len(pred_labels)):
        if pred_labels[i] <> 'O' and label_y[i] <> 'O' and pred_labels[i] == label_y[i]:
            tp += 1
        elif pred_labels[i] == 'O' and label_y[i] <> 'O':
            fn += 1
        elif pred_labels[i] <> 'O' and label_y[i] <> pred_labels[i]:
            fp += 1
prec = tp/(tp+fp)
rec = tp/(tp+fn)
fmes = 2*(prec*rec)/(prec+rec)
print ("True Positives: {:.1f}".format(tp))
print ("False Positives: {:.1f}".format(fp))
print ("False Negatives: {:.1f}".format(fn))
print ("Precision: {:.6f}".format(prec))
print ("Recall: {:.6f}".format(rec))
print ("F-Measure: {:.6f}".format(fmes))
print ("-----------------------------------------------\n")
exit(0)
'''
print 'Training...'

for epoch in xrange(number_of_epochs):
    print 'Epoch '+str(number_of_epochs+1)
    start_time = time.time()
    
    #Train for 1 epoch
    hist = model.fit(train_x, train_y_cat, nb_epoch=1, batch_size=minibatch_size, verbose=True, shuffle=True,show_accuracy=True,)
                     #validation_data=(dev_x,dev_y))
    #hist = model.fit(train_x, train_y_cat, batch_size=minibatch_size, verbose=True, shuffle=True)
    #validation = model.evaluate(dev_x, dev_y, batch_size=dev_x.shape[0], show_accuracy=True)
    print "%.2f sec for training\n" % (time.time() - start_time)
    '''
    ### Testing ###
    tp = 0.0
    fp = 0.0
    fn = 0.0

    for batch in iterate_minibatches(dev_x, dev_y, 100, shuffle=False):
        inputs, a = batch
        b = model.predict_classes(inputs, verbose=0)
        label_y = []
        pred_labels = []
        for sentence in a:
            for element in sentence:
                label_y.append(idx2Label[np.argmax(element)])
        for sentence in b:
            for element in sentence:
                pred_labels.append(idx2Label[element])
        for i in xrange(0,len(pred_labels)):
            if pred_labels[i] <> 'O' and label_y[i] <> 'O' and pred_labels[i] == label_y[i]:
                tp += 1
            elif pred_labels[i] == 'O' and label_y[i] <> 'O':
                fn += 1
            elif pred_labels[i] <> 'O' and label_y[i] <> pred_labels[i]:
                fp += 1
    try:
        prec = tp/(tp+fp)
        rec = tp/(tp+fn)
        fmes = 2*(prec*rec)/(prec+rec)
        print ("True Positives: {:.1f}".format(tp))
        print ("False Positives: {:.1f}".format(fp))
        print ("False Negatives: {:.1f}".format(fn))
        print ("Precision: {:.6f}".format(prec))
        print ("Recall: {:.6f}".format(rec))
        print ("F-Measure: {:.6f}".format(fmes))
        print ("-----------------------------------------------\n")
    except:
        print ("Erro de divisão por zero. Continuando...")
    '''
    ### Testing ###
    tp = 0.0
    fp = 0.0
    fn = 0.0

    for batch in iterate_minibatches(dev_x, dev_y, 2000, shuffle=False):
        inputs, a = batch
        b = model.predict_classes(inputs, verbose=0)
        label_y = [idx2Label[element] for element in a]
        pred_labels = [idx2Label[element] for element in b]
        for i in xrange(0,len(label_y)):
            if pred_labels[i] <> 'O' and label_y[i] <> 'O' and pred_labels[i] == label_y[i]:
                tp += 1
            elif pred_labels[i] == 'O' and label_y[i] <> 'O':
                fn += 1
            elif pred_labels[i] <> 'O' and label_y[i] <> pred_labels[i]:
                fp += 1
    try:
        prec = tp/(tp+fp)
        rec = tp/(tp+fn)
        fmes = 2*(prec*rec)/(prec+rec)
        print ("True Positives: {:.1f}".format(tp))
        print ("False Positives: {:.1f}".format(fp))
        print ("False Negatives: {:.1f}".format(fn))
        print ("Precision: {:.6f}".format(prec))
        print ("Recall: {:.6f}".format(rec))
        print ("F-Measure: {:.6f}".format(fmes))
        print ("-----------------------------------------------\n")
    except:
        print ("Erro de divisão por zero. Continuando...")
    # Compute precision, recall, F1 on dev & test data
    #pre_dev, rec_dev, f1_dev = BIOF1Validation.compute_f1(model.predict_classes(dev_x, verbose=0), dev_y, idx2Label)
    #pre_test, rec_test, f1_test = BIOF1Validation.compute_f1(model.predict_classes(test_x, verbose=0), test_y, idx2Label)

    #print "%d epoch: F1 on dev: %f, F1 on test: %f" % (epoch+1, f1_dev, f1_test)'''
    model.save_weights('modelos_keras/parciais/modelo_keras'+str(epoch)+'.h5',overwrite=True)

model.save_weights('modelos_keras/modelo_keras.h5')


#model.load_weights('modelos_keras/parciais/modelo_keras83.h5')