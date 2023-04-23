# Importing Libraries
import pickle
import re
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, BatchNormalization, Bidirectional, LSTM, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# File Names
GLOVE_EMBEDDING_FILE_NAME = '../data/glove.6B/glove.6B.300d.txt'

TRAIN_FILE_NAME = None
DEV_FILE_NAME = None
TEST_FILE_NAME = None

EMBEDDINGS_FILE_NAME = None
CHAR2IDX_FILE_NAME = None
TRAIN_PREPROCESS_FILE_NAME = None
TEST_PREPROCESS_FILE_NAME = None

MODEL_FILE_NAME = None
RESULT_FILE_NAME = None


# Hyperparameters
EMBEDDING_DIMENSION = 300
MAX_SENTENCE_LEN = 50
LSTM_UNITS = 64
HIDDEN_DIM = 600
CLASSES = 3 
BATCH_SIZE = 512
TRAINING_EPOCHS = 50
DROPOUT = 0.2
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.02


# Save file to pickle
def save(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)


# load file from pickle
def load(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data


# Write Results to file
def write_results_to_file(sentence1, sentence2, actual_labels, predicted_labels, test_accuracy) :
    # Mapping the numerical representation of tag to corresponding tag
    labels = {
        0: 'entailment',
        1: 'neutral',
        2: 'contradiction'
    }
    with open(RESULT_FILE_NAME, 'w', encoding = 'utf-8') as file :
        file.write(f'Test Accuracy : {test_accuracy:1.4f}\n')
        file.write(f'-----------------------------------------------------------\n')
        file.write(f'Predicted_Labels || Actual_Labels || Sentence1 || Sentence1\n')
        file.write(f'-----------------------------------------------------------\n')
        for sent1, sent2, actual_label, predicted_label in zip(sentence1, sentence2, actual_labels, predicted_labels) :
            file.write(f'{labels[predicted_label]} || {labels[actual_label]} || {sent1} || {sent2}\n')


# Processing the sentence.
def process_sentence(sentence) :
    sentence = re.sub('[^0-9a-zA-Z\'\s]+', '', sentence)
    sentence = sentence.lower()
    sentence = ''.join(ch if ch != ' ' else '' for ch in sentence) 
    return sentence


# Basic Preprocessing of file.
def preprocess_file(filename):
    print(f'Doing Basic Preprocessing of File {filename}.')

    # Reading the file
    dataframe = pd.read_csv(filename, sep="\t", encoding = 'utf-8', on_bad_lines = 'skip')

    # Extracting below 3 columns
    dataframe = dataframe[['gold_label', 'sentence1', 'sentence2']]

    # Dropping records which doesn't have label, or column have None value
    dataframe = dataframe[dataframe['gold_label'] != '-']
    dataframe = dataframe.dropna()

    # Mapping the corresponding tag to numerical representation
    labels = {
        'entailment': 0,
        'neutral': 1,
        'contradiction': 2
    }
    dataframe['gold_label'] = dataframe['gold_label'].map(labels)

    # lowercasing sentences
    dataframe['sentence1'] = dataframe['sentence1'].apply(process_sentence)
    dataframe['sentence2'] = dataframe['sentence2'].apply(process_sentence)

    #  Converting lables, and sentences to array
    labels_data = dataframe['gold_label'].values
    sentence1_data = dataframe['sentence1'].values
    sentence2_data = dataframe['sentence2'].values

    return labels_data, sentence1_data, sentence2_data


# Building char2idx dictionary
def build_char2idx(sentence1_data, sentence2_data) :
    char2idx = {}
    char2idx[' '] = 0
    idx = 1
    for i in tqdm(range(sentence1_data.shape[0]), desc = 'Building char2idx Dictionary'):
        for ch in sentence1_data[i] :
            if ch not in char2idx :
                char2idx[ch] = idx
                idx += 1

        for ch in sentence2_data[i] :
            if ch not in char2idx :
                char2idx[ch] = idx
                idx += 1
    
    return char2idx


# Building char embeddings from pre trained embeddings
def build_embedding_matrix(char2idx):
    print('Building Word Embeddings.')
    word_embeddings = {}
    idx = 0
    with open(GLOVE_EMBEDDING_FILE_NAME, 'r', encoding = 'utf-8') as file:
        for line in file :
            word, embedding_vector = line[:-1].split(' ')[0], line[:-1].split(' ')[1:]
            word_embeddings[word] = np.array(embedding_vector, dtype = np.float32)
            idx += 1
            print(f'lines readed : {idx}\r', end = '')
    print()

    embedding_matrix = []
    for ch in char2idx.keys() :
        if ch in word_embeddings.keys() :
            embedding_matrix.append(word_embeddings[ch])
        else :
            embedding_matrix.append(np.zeros(EMBEDDING_DIMENSION, dtype=np.float32))
    embedding_matrix = np.array(embedding_matrix)
    return embedding_matrix


# vectorising sentence
def vectorise_sentence(sentence, char2idx) :
    vectorised_sentence = [0] * (MAX_SENTENCE_LEN - len(sentence))
    for ch in sentence :
        vectorised_sentence.append(char2idx.get(ch, 0))
    return np.array(vectorised_sentence[:MAX_SENTENCE_LEN])


# vectorising list of sentences
def vectorise_sentences(sentence_data, char2idx) :
    vectorised_sentences = []
    for sentence in tqdm(sentence_data, desc = 'Vectorising Sentence') :
        vectorised_sentences.append(vectorise_sentence(sentence, char2idx))
    return np.array(vectorised_sentences)


# Preprocessing part
def handle_preprocess_part() : 
    # preprocessing the file
    train_labels_data, train_sentence1_data, train_sentence2_data = preprocess_file(TRAIN_FILE_NAME)
    dev_labels_data, dev_sentence1_data, dev_sentence2_data = preprocess_file(DEV_FILE_NAME)
    test_labels_data, test_sentence1_data, test_sentence2_data = preprocess_file(TEST_FILE_NAME)

    train_labels_data = np.concatenate((train_labels_data, dev_labels_data))
    train_sentence1_data = np.concatenate((train_sentence1_data, dev_sentence1_data))
    train_sentence2_data = np.concatenate((train_sentence2_data, dev_sentence2_data))

    # Building char2idx dictionary
    char2idx = build_char2idx(train_sentence1_data, train_sentence2_data)

    # Building embedding_matrix
    embedding_matrix = build_embedding_matrix(char2idx)

    # Vectorising sentences
    train_sentence1_vectorised = vectorise_sentences(train_sentence1_data, char2idx)
    train_sentence2_vectorised = vectorise_sentences(train_sentence2_data, char2idx)

    test_sentence1_vectorised = vectorise_sentences(test_sentence1_data, char2idx)
    test_sentence2_vectorised = vectorise_sentences(test_sentence2_data,char2idx)

    print('doing one hot encoding of labels.')
    # One hot Encoding of Labels
    lb = LabelBinarizer()
    lb.fit([0, 1, 2])
    train_labels_encoded = lb.transform(train_labels_data)
    test_labels_encoded = lb.transform(test_labels_data)

    print('Saving the Preprocessed Data.')
    # Saving the Preprocessed Data
    save((train_sentence1_vectorised, train_sentence2_vectorised, train_labels_encoded), TRAIN_PREPROCESS_FILE_NAME)
    save((test_sentence1_vectorised, test_sentence2_vectorised, test_labels_encoded), TEST_PREPROCESS_FILE_NAME)
    save(embedding_matrix, EMBEDDINGS_FILE_NAME)
    save(char2idx, CHAR2IDX_FILE_NAME)


# Building the Model
def build_BL_model(embedding_matrix, embedding_dim, lstm_units, hidden_dim, classes, max_seq_len, dropout, learning_rate):
    # Define the embedding layer with the pre trained embedding_matrix
    embedding = Embedding(input_dim=embedding_matrix.shape[0],
                                output_dim=embedding_dim,
                                weights=[embedding_matrix],
                                input_length=max_seq_len,
                                trainable=True)

    # Define the bidirectional LSTM layer
    BiLSTM = Bidirectional(LSTM(lstm_units))

    # Define the input layers and its shapes for premise and hypothesis
    premise = Input(shape = (max_seq_len), dtype='int32')
    hypothesis = Input(shape = (max_seq_len), dtype='int32')

    # Embed the premise and hypothesis
    premise_embedded = embedding(premise)
    hypothesis_embedded = embedding(hypothesis)

    # Apply the bidirectional LSTM layer
    premise_BiLSTM = BiLSTM(premise_embedded)
    hypothesis_BiLSTM = BiLSTM(hypothesis_embedded)

    # Apply Batch normalization
    premise_normalized = BatchNormalization()(premise_BiLSTM)
    hypothesis_normalized = BatchNormalization()(hypothesis_BiLSTM)

    # Concatenate the normalized premise and hypothesis and apply a dropout layer
    train_input = concatenate([premise_normalized, hypothesis_normalized])
    train_input = Dropout(dropout)(train_input)

    # Apply the (Dense layer, Dropout layer. Batch normalization layer)
    train_input = Dense(hidden_dim, activation='relu')(train_input)
    train_input = Dropout(DROPOUT)(train_input)
    train_input = BatchNormalization()(train_input)

    # Define the output Dense layer
    prediction = Dense(classes, activation='softmax')(train_input)

    # Define the complete model
    model = Model(inputs=[premise, hypothesis], outputs=prediction)

    # Choosing an optimizer
    optimizer = Adam(learning_rate=learning_rate)

    # Compile the model and print out the model summary
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    return model


# Training Model
def train_model(model, X1, X2, y, batch_size, epochs, validation_split, model_file_name) :
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    model_checkpoint = ModelCheckpoint(model_file_name, save_best_only = True)

    callbacks = [early_stopping, model_checkpoint]

    # Train the model
    history = model.fit(x=[X1, X2],
                        y=y,
                        batch_size=BATCH_SIZE,
                        epochs=TRAINING_EPOCHS,
                        validation_split=VALIDATION_SPLIT,
                        callbacks=callbacks)


# Loading Test sentences
def load_test_sentences(filename) :
    # Reading the file
    dataframe = pd.read_csv(filename, sep="\t", encoding = 'utf-8', on_bad_lines = 'skip')
    
    # Extracting below 3 columns
    dataframe = dataframe[['gold_label', 'sentence1', 'sentence2']]
    
    # Dropping records which doesn't have label, or column have None value
    dataframe = dataframe[dataframe['gold_label'] != '-']
    dataframe = dataframe.dropna()
    
    sentence1_data = dataframe['sentence1'].values
    sentence2_data = dataframe['sentence2'].values
    return sentence1_data, sentence2_data


# Train Model
def handle_train_part():
    print('Loading the Preprocessed Data.')
    # Loading the Preprocessed Data
    train_sentence1_vectorised, train_sentence2_vectorised, train_labels_encoded = load(TRAIN_PREPROCESS_FILE_NAME)
    embedding_matrix = load(EMBEDDINGS_FILE_NAME)
    print(train_sentence1_vectorised.shape, embedding_matrix.shape)

    # Building Model
    model = build_BL_model(embedding_matrix=embedding_matrix,
                           embedding_dim=EMBEDDING_DIMENSION,
                           lstm_units=LSTM_UNITS,
                           hidden_dim=HIDDEN_DIM,
                           classes=CLASSES,
                           max_seq_len=MAX_SENTENCE_LEN,
                           dropout=DROPOUT,
                           learning_rate=LEARNING_RATE)

    # Train Model
    train_model(model = model,
                X1=train_sentence1_vectorised,
                X2=train_sentence2_vectorised,
                y=train_labels_encoded,
                batch_size=BATCH_SIZE,
                epochs=TRAINING_EPOCHS,
                validation_split=VALIDATION_SPLIT,
                model_file_name=MODEL_FILE_NAME)


# For Testing Part
def handle_test_part() :
    # Loading Test Data
    print('Loading Test Data')
    sentence1_data, sentence2_data = load_test_sentences(TEST_FILE_NAME)
    test_sentence1_vectorised, test_sentence2_vectorised, test_labels_encoded = load(TEST_PREPROCESS_FILE_NAME)
    print(test_sentence1_vectorised.shape)
    
    # Loading Model
    print('Loading Model')
    model = load_model(MODEL_FILE_NAME)

    # Probability Prediction
    predicted_prob = model.predict(
        x = [test_sentence1_vectorised, test_sentence2_vectorised],
        batch_size=BATCH_SIZE
    )

    # Predicting Labels
    test_labels = test_labels_encoded.argmax(axis = 1)
    test_labels_predicted = predicted_prob.argmax(axis = 1)

    test_accuracy = accuracy_score(test_labels, test_labels_predicted)

    print(f'Writing Results to File : {RESULT_FILE_NAME}')
    write_results_to_file(sentence1_data, sentence2_data, test_labels, test_labels_predicted, test_accuracy)

    print("-------------------------------------------------------")
    print(f'Test Accuracy : {test_accuracy:1.4f}')
    print("-------------------------------------------------------")
    print('Classification Report : \n')
    print(classification_report(test_labels, test_labels_predicted, digits = 4, target_names = ['Entailment', 'Neutral', 'Contradiction']))
    print("-------------------------------------------------------")
    print('Confusion Matrix : \n')
    cm = confusion_matrix(test_labels, test_labels_predicted, labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Entailment', 'Neutral', 'Contradiction'])
    disp.plot()
    plt.show()


# for inference part from 2 input sentences.
def handle_inference_part():
    print("-------------------------------------------------------")
    sentence1 = input('Enter Sentence1 : ')
    sentence2 = input('Enter Sentence2 : ')

    # Loading char2idx dictionary
    char2idx = load(CHAR2IDX_FILE_NAME)
    
    # Loading the model
    model = load_model(MODEL_FILE_NAME)

    # Vectorizing sentences
    sentence1_processed = process_sentence(sentence1)
    sentence2_processed = process_sentence(sentence2)

    sentence1_vectorised = vectorise_sentences([sentence1_processed], char2idx)
    sentence2_vectorised = vectorise_sentences([sentence2_processed], char2idx)

    # Predicting
    predicted_prob = model.predict(x = [sentence1_vectorised, sentence2_vectorised])
    label_predicted = predicted_prob.argmax(axis = 1)

    # Mapping the numerical representation of tag to corresponding tag
    labels = {
        0: 'entailment',
        1: 'neutral',
        2: 'contradiction'
    }

    print("-------------------------------------------------------")
    print(f'sentence1 : {sentence1}')
    print(f'sentence2 : {sentence2}')
    print(f'Predicted Label : {labels[label_predicted[0]]}')
    print("-------------------------------------------------------")


# initialising File Names
def init_file_names(dataset_name) :
    global TRAIN_FILE_NAME, DEV_FILE_NAME, TEST_FILE_NAME
    global EMBEDDINGS_FILE_NAME, CHAR2IDX_FILE_NAME, TRAIN_PREPROCESS_FILE_NAME, TEST_PREPROCESS_FILE_NAME
    global  MODEL_FILE_NAME, RESULT_FILE_NAME
    
    TRAIN_FILE_NAME = f'../data/{dataset_name}_1.0_train.txt'
    DEV_FILE_NAME = f'../data/{dataset_name}_1.0_dev.txt'
    TEST_FILE_NAME = f'../data/{dataset_name}_1.0_test.txt'

    EMBEDDINGS_FILE_NAME = f'./PreProcessed_data/char_emeddings_{dataset_name}.pickle'
    CHAR2IDX_FILE_NAME = f'./PreProcessed_data/char2idx_{dataset_name}.pickle'
    TRAIN_PREPROCESS_FILE_NAME = f'./PreProcessed_data/train_preprocessed_{dataset_name}.pickle'
    TEST_PREPROCESS_FILE_NAME = f'./PreProcessed_data/test_preprocessed_{dataset_name}.pickle'

    MODEL_FILE_NAME = f'./Models/BiLSTM_Char_Level_Model_{dataset_name}.h5'
    RESULT_FILE_NAME = f'./Results/BiLSTM_Char_Level_Results_{dataset_name}.txt'


# Driver Code
if __name__ == '__main__':
    
    # Dataset Selection
    print("-------------------------------------------------------")
    print("Which Dataset you want to use ?")
    print("1. SNLI")
    print("2. MULTINI")
    ch1 = int(input('Enter you choice : '))

    if ch1 == 1:  # snli
        init_file_names('snli')

    elif ch1 == 2:  # multinli
        init_file_names('multinli')

    else:
        print("Invalid Input.")
        exit()

    # Operation Selection
    print("-------------------------------------------------------")
    print("Which Operation you want to perform ?")
    print("1. Preprocess Data")
    print("2. Train Model")
    print("3. Test Model on Test file")
    print("4. Test Model on Manual Sentences")
    ch = int(input('Enter you choice : '))
    
    if ch == 1:  # preprocess
        handle_preprocess_part()

    elif ch == 2:  # train
        handle_train_part()

    elif ch == 3:  # test
        handle_test_part()

    elif ch == 4:  # inference
        handle_inference_part()

    else:
        print("Invalid Input.")
