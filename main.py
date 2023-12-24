#Import Library
###############
import string
import re
import tensorflow
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical,pad_sequences
from keras.utils.vis_utils import plot_model
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Embedding, RepeatVector, TimeDistributed
from keras.callbacks import ModelCheckpoint
from pickle import dump, load
from unicodedata import normalize
from numpy import array, argmax
from numpy.random import rand, shuffle
from nltk.translate.bleu_score import corpus_bleu

#Loading the Document
#####################
#filename = 'E:\ML Projects\Machine Translation\dataset\deu.txt'

#Clean Text
###########
#load the document into memory
def load_doc(filename):
    file = open(filename, mode = 'rt', encoding = 'utf-8')
    text = file.read()
    file.close()
    return text

#split the loaded document into sentences
def to_pairs(doc):
    lines = doc.strip().split('\n')
    pairs = [line.split('\t') for line in lines]
    return pairs

#clean a list of lines
def clean_pairs(lines):
    cleaned = list()
    re_print = re.compile('[^%s]' % re.escape(string.printable))
    table = str.maketrans('','',string.punctuation)
    for pair in lines:
        clean_pair = list()
        for line in pair:
            line = normalize ('NFD', line).encode('ascii', 'ignore')
            line = line.decode('UTF-8')
            line = line.split()
            line = [word.lower() for word in line]
            line = [word.translate(table) for word in line]
            line = [re_print.sub('', w) for w in line]
            line = [word for word in line if word.isalpha()]
            clean_pair.append(' '.join(line))
            cleaned.append(clean_pair)
    return array(cleaned)

#save list of clean sentences into files
def save_clean_data(sentences, filename):
    dump(sentences, open(filename, 'wb'))
    print('Saved: %s' % filename)

#load dataset
filename = 'E:\ML Projects\Machine Translation\dataset\deu.txt'
doc = load_doc(filename)
#split english-german into pairs
pairs = to_pairs(doc)
#clean sentences
clean_pairs = clean_pairs(pairs)
#save file
save_clean_data(clean_pairs, 'english-german.pkl')
for i in range(100):
    print('[%s] => [%s]' % (clean_pairs[i,0], clean_pairs[i,1]))

#Split Text
###########
#load clean dataset
def load_clean_sentences(filename):
    return load(open(filename, 'rb'))

#save list of clean sentences into files
def save_clean_data(sentences, filename):
    dump(sentences, open(filename, 'wb'))
    print('Saved: %s' % filename)

#load dataset
raw_dataset = load_clean_sentences('english-german.pkl')

#dataset size reduction
n_sentences = 10000
dataset = raw_dataset[:n_sentences, :]

#shuffle the dataset
shuffle(dataset)

#split train and test dataset
train, test = dataset [:9000], dataset[9000:]

#save clean data
save_clean_data(dataset, 'english-german-both.pkl')
save_clean_data(train, 'english-german-train.pkl')
save_clean_data(test, 'english-german-test.pkl')

#Train Neutral Translation Model
#################################
#load clean dataset
def load_clean_sentences(filename):
    return load(open(filename,'rb'))

#load datasets
dataset = load_clean_sentences()
train = load_clean_sentences()
test = load_clean_sentences()

#fit a tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

#maximum sentence length
def max_length(lines):
    return max(len(line.split()) for line in lines)

#prepare english tokenizer
eng_tokenizer = create_tokenizer(dataset[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_length = max_length(dataset[:,0])
print('English Vocabulary Size: %d' % eng_vocab_size)
print('English Max Length %d' % (eng_length))
#prepare german tokenizer
ger_tokenizer = create_tokenizer(dataset[:, 1])
ger_vocab_size = len(ger_tokenizer.word_index) + 1
ger_length = max_length(dataset[:,1])
print('German Vocabulary Size: %d' % ger_vocab_size)
print('German Max Length %d' % (ger_length))

#encode and pad sequences
def encode_sequences(tokenizer, length, lines):
    X = tokenizer.texts_to_sequences(lines)
    X = pad_sequences(X, maxlen = length, padding = 'post')
    return X

#encode target sequences
def encode_output(sequences, vocab_size):
    ylist = list()
    for sequence in sequences:
        encoded = to_categorical(sequence, num_classes=vocab_size)
        ylist.append(encoded)
    y = array(ylist)
    y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
    return y

#prepare training data
trainX = encode_sequences(ger_tokenizer, ger_length, train[:, 1])
trainY = encode_sequences(eng_tokenizer, eng_length, train[:, 0])
trainY = encode_output(trainY, eng_vocab_size)
#prepare validation data
testX = encode_sequences(ger_tokenizer, ger_length, train[:, 1])
testY = encode_sequences(eng_tokenizer, eng_length, train[:, 0])
testY = encode_output(testY, eng_vocab_size)

#define NMT model
def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
    model = Sequential()
    model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
    model.add(LSTM(n_units))
    model.add(RepeatVector(tar_timesteps))
    model.add(LSTM(n_units, return_sequences = True))
    model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
    return model

#define model
model = define_model(ger_vocab_size, eng_vocab_size, ger_length, eng_length, 256)
model.compile(optimizer='adam', loss='categorical_crossentropy')
#summaize defined model
print(model.summary())
plot_model(model,to_file='model.png',show_shapes=True)

#fit model [process]
filename = 'model.h5'
checkpoint = ModelCheckpoint(filename, monitor = 'val_loss', verbose = 1, save_best_only=True, mode='min')
model.fit(trainX, trainY, epochs=30, batch_size=64, validation_data=(testX,testY),callbacks=[checkpoint], verbose=2)

#Evaluate Neural Translation Model
##################################
#load a clean data set
def load_clean_sentences(filename):
    return load(open(filename,'rb'))

#fit a tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

#maximum sentence length
def max_length(lines):
    return max(len(line.split()) for line in lines)

#encode and pad sequences
def encode_sequences(tokenizer, length, lines):
    X = tokenizer.texts_to_sequences(lines)
    X = pad_sequences(X, maxlen = length, padding = 'post')
    return X

#map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
        return None

#generate target given source sequence
def predict_sequence(model, tokenizer, source):
    prediction = model.predict(source, verbose = 0)[0]
    integers = [argmax(vector) for vector in prediction]
    target = list()
    for i in integers:
        word = word_for_id(i,tokenizer)
        if word is None:
            break
        target.append(word)
    return ' '.join(target)

#evaluate the model's skill
def evaluate_model(model, tokenizer, sources, raw_dataset):
    actual, predicted = list(), list()
    for i, source in enumerate(sources):
        #translate encoded source text
        source = source.reshape((1, source.shape[0]))
        translation = predict_sequence(model, eng_tokenizer, source)
        raw_target, raw_src = raw_dataset[i]
        if i < 10:
            print('src=[%s], target =[%s], predicted=[%s]' % (raw_src, raw_target, translation))
            actual.append([raw_target.split()])
            predicted.append(translation.split())
        #calculate and print out BLEU source
        print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0,0,0,0)))
        print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
        print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
        print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))


#load datasets for 3 different variables, dataset, train and test
dataset = load_clean_sentences('english-german-both.pkl')
train = load_clean_sentences('english-german-train.pkl')
test = load_clean_sentences('english-german-test.pkl')
#prepare english tokenizer
eng_tokenizer = create_tokenizer(dataset[:,0])
eng_vocab_size = len(eng_tokenizer.word_index)+1
eng_length = max_length(dataset[:,0])
#prepare german tokenizer
ger_tokenizer = create_tokenizer(dataset[:,1])
ger_vocab_size = len(ger_tokenizer.word_index) + 1
ger_length = max_length(dataset[:,1])
#prepare data
trainX = encode_sequences(ger_tokenizer,ger_length, train[:,1])
testX = encode_sequences(ger_tokenizer, ger_length, test[:,1])

#load model
model = load_model('model.h5')
#test on some training sequences
print('train')
evaluate_model(model, eng_tokenizer, trainX, train)
#test on some test sequences
print('test')
evaluate_model(model, eng_tokenizer, textX, test)

#set up translation variable
evaluate_model(load_model('model.h5'), eng_tokenizer, testX, test[:,:2])
