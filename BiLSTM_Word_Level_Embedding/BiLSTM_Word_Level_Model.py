# Importing Libraries
import pickle
import re
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# File Names
TRAIN_FILE_NAME = None
DEV_FILE_NAME = None
TEST_FILE_NAME = None

WORD2IDX_FILE_NAME = None
TRAIN_PREPROCESS_FILE_NAME = None
DEV_PREPROCESS_FILE_NAME = None
TEST_PREPROCESS_FILE_NAME = None

MODEL_FILE_NAME = None
RESULT_FILE_NAME = None


# Hyperparameters
EMBEDDING_DIM = 100
HIDDEN1_DIM = 100
HIDDEN2_DIM = 100
HIDDEN3_DIM = 10
CLASSES = 3
MAX_SENTENCE_LEN = 40
BATCH_SIZE = 128
LEARNING_RATE = 0.001
TRAINING_EPOCHS = 20


# Checking if cuda available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on Device : {DEVICE}")


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


# Building word2idx dictionary
def build_word2idx(sentence1_data, sentence2_data) :
    word2idx = {}
    idx = 1
    for i in tqdm(range(sentence1_data.shape[0]), desc = 'Building word2idx Dictionary'):
        for word in sentence1_data[i].split() :
            if word not in word2idx :
                word2idx[word] = idx
                idx += 1

        for word in sentence2_data[i].split() :
            if word not in word2idx :
                word2idx[word] = idx
                idx += 1
    
    return word2idx
    

# vectorising sentence
def vectorise_sentence(sentence, word2idx) :
    vectorised_sentence = [word2idx.get(word, 0) for word in sentence.split() if word.strip() != '' ]
    vectorised_sentence += [0] * (MAX_SENTENCE_LEN - len(vectorised_sentence))
    return vectorised_sentence[:MAX_SENTENCE_LEN]
    

# vectorising list of sentences
def vectorise_sentences(sentence_data, word2idx) :
    vectorised_sentences = []
    for sentence in tqdm(sentence_data, desc = 'Vectorising Sentence') :
        vectorised_sentences.append(vectorise_sentence(sentence, word2idx))
    return vectorised_sentences


# Preprocessing part
def handle_preprocess_part() : 
    # preprocessing the file
    train_labels_data, train_sentence1_data, train_sentence2_data = preprocess_file(TRAIN_FILE_NAME)
    dev_labels_data, dev_sentence1_data, dev_sentence2_data = preprocess_file(DEV_FILE_NAME)
    test_labels_data, test_sentence1_data, test_sentence2_data = preprocess_file(TEST_FILE_NAME)

    # Building word2idx dictionary
    word2idx = build_word2idx(train_sentence1_data, train_sentence2_data)

    # Vectorising sentences
    train_sentence1_vectorised = vectorise_sentences(train_sentence1_data, word2idx)
    train_sentence2_vectorised = vectorise_sentences(train_sentence2_data, word2idx)

    dev_sentence1_vectorised = vectorise_sentences(dev_sentence1_data, word2idx)
    dev_sentence2_vectorised = vectorise_sentences(dev_sentence2_data, word2idx)

    test_sentence1_vectorised = vectorise_sentences(test_sentence1_data, word2idx)
    test_sentence2_vectorised = vectorise_sentences(test_sentence2_data,word2idx)

    # Coverting to Tensor
    train_sentence1_vectorised = torch.LongTensor(train_sentence1_vectorised)
    train_sentence2_vectorised = torch.LongTensor(train_sentence2_vectorised)
    train_labels_data = torch.LongTensor(train_labels_data)
    
    dev_sentence1_vectorised = torch.LongTensor(dev_sentence1_vectorised)
    dev_sentence2_vectorised = torch.LongTensor(dev_sentence2_vectorised)
    dev_labels_data = torch.LongTensor(dev_labels_data)
    
    test_sentence1_vectorised = torch.LongTensor(test_sentence1_vectorised)
    test_sentence2_vectorised = torch.LongTensor(test_sentence2_vectorised)
    test_labels_data = torch.LongTensor(test_labels_data)

    print('Saving the Preprocessed Data.')
    # Saving the Preprocessed Data
    save((train_sentence1_vectorised, train_sentence2_vectorised, train_labels_data), TRAIN_PREPROCESS_FILE_NAME)
    save((dev_sentence1_vectorised, dev_sentence2_vectorised, dev_labels_data), DEV_PREPROCESS_FILE_NAME)
    save((test_sentence1_vectorised, test_sentence2_vectorised, test_labels_data), TEST_PREPROCESS_FILE_NAME)
    save(word2idx, WORD2IDX_FILE_NAME)


# class dataset
class NliDataset(Dataset):
    def __init__(self, premise, hypothesis, labels):
        self.premise = premise
        self.hypothesis = hypothesis
        self.labels = labels

    def __getitem__(self, index):
        premise_seq = self.premise[index]
        hypothesis_seq = self.hypothesis[index]
        label = self.labels[index]
        return premise_seq, hypothesis_seq, label

    def __len__(self):
        return len(self.labels)


# BiLSTM Class
class BiLSTM_Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden1_dim, hidden2_dim, hidden3_dim, classes):
        super(BiLSTM_Model, self).__init__()

        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM Layer
        self.lstm = nn.LSTM(embedding_dim, hidden1_dim, bidirectional=True, batch_first=True)
        
        # Linear Layers
        self.linear1 = nn.Linear(2 * hidden1_dim, hidden2_dim)
        self.linear2 = nn.Linear(hidden2_dim, hidden3_dim)
        self.linear3 = nn.Linear(hidden3_dim, classes)

        # Activation Layer
        self.relu = nn.ReLU()

    def forward(self, premise, hypothesis):
        premise_embedded = self.embedding(premise)
        hypothesis_embedded = self.embedding(hypothesis)

        _, (x1,_) = self.lstm(premise_embedded)
        _, (x2,_) = self.lstm(hypothesis_embedded)
        x1 = x1[-1,:,:]
        x2 = x2[-1,:,:]
        x = torch.cat((x1, x2), dim=1)

        x = self.linear1(x)
        x = self.relu(x)

        x = self.linear2(x)
        x = self.relu(x)

        x = self.linear3(x)
        x = self.relu(x)
        return x


# Training
def train(model, dataloader, optimizer, loss_fn) :
    # Set model to training mode
    model.train()

    # Initialize counters for loss and accuracy
    total_loss = 0
    count_correct = 0
    count_total = 0
    
    # Iterate over all input sequences
    for idx, (batch_premise, batch_hypothesis, batch_labels) in tqdm(enumerate(dataloader), total = len(dataloader), desc = 'Training') :
        
        # Moving data to GPU
        batch_premise = batch_premise.to(DEVICE)
        batch_hypothesis = batch_hypothesis.to(DEVICE)
        batch_labels = batch_labels.to(DEVICE)

        # Compute output of model for the input premise, and hypothesis
        model_output = model(batch_premise, batch_hypothesis)

        # Compute loss between predicted and target tags
        loss = loss_fn(model_output, batch_labels)
        total_loss += loss.cpu().item()
        
        # Compute number of correct predictions and total number of predictions
        predicted_labels = torch.argmax(model_output, 1)
        count_correct += torch.sum(predicted_labels == batch_labels).cpu()
        count_total += batch_labels.shape[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Compute average loss and accuracy over all input premise, and hypothesis
    loss =  total_loss / len(dataloader)
    accuracy = (count_correct/count_total).item()  
    return loss, accuracy


# Testing and Evaluating Loss, Accuracy
def test( model, dataloader, loss_fn) :
    # Set model to eval mode
    model.eval()

    # Disable gradient computation
    with torch.no_grad() :
        # Initialize counters for loss and accuracy
        total_loss = 0
        count_correct = 0
        count_total = 0

        # Iterate over all input sequences
        for idx, (batch_premise, batch_hypothesis, batch_labels) in tqdm(enumerate(dataloader), total = len(dataloader), desc = 'Evaluating') :
            
            # Moving data to GPU
            batch_premise = batch_premise.to(DEVICE)
            batch_hypothesis = batch_hypothesis.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)

            # Compute output of model for the input premise, and hypothesis
            model_output = model(batch_premise, batch_hypothesis)

            # Compute loss between predicted and target tags
            loss = loss_fn(model_output, batch_labels)
            total_loss += loss.cpu().item()
            
            # Compute number of correct predictions and total number of predictions
            predicted_labels = torch.argmax(model_output, 1)
            count_correct += torch.sum(predicted_labels == batch_labels).cpu()
            count_total += batch_labels.shape[0]

    # Compute average loss and accuracy over all input premise, and hypothesis
    loss =  total_loss / len(dataloader)
    accuracy = (count_correct/count_total).item()  

    return loss, accuracy


# fitting the model on train data
def fit(model, train_dataloader, dev_dataloader, optimizer, loss_fn) :

    # defining the best loss value, and patience(for early stopping)
    best_val_loss = float('inf')
    PATIENCE = 6
    CONTROL = 0

    for epoch_num in range(TRAINING_EPOCHS) :
        print('------------------------------------------------------------------------')
        print(f"Epoch : {epoch_num + 1}")

        # training the model on the training data
        train_loss, train_accuracy = train(model, train_dataloader, optimizer, loss_fn)
        
        # evaluating the model on the validation data
        val_loss, val_accuracy = test(model, dev_dataloader, loss_fn)
        print(f'Train Loss: {train_loss}\t Train Accuracy: {train_accuracy}')
        print(f'Val. Loss: {val_loss}\t Val. Accuracy: {val_accuracy}')

        # saving the model if the current validation loss is the best seen so far
        if val_loss < best_val_loss :
            best_val_loss = val_loss
            CONTROL = 0
            torch.save(model, MODEL_FILE_NAME)
            print(f'Best Model Saved as {MODEL_FILE_NAME} with val_loss: {val_loss}, and val_accuracy: {val_accuracy}.')
        
            # if the validation loss has not improved, increment CONTROL
        elif CONTROL < PATIENCE:
            CONTROL += 1
        
        # if CONTROL has reached PATIENCE, stop training early
        else :
            break
    print('------------------------------------------------------------------------')


# For Training Part
def handle_train_part():

    # Loading the Preprocessed Data
    print('Loading the Preprocessed Data.')
    train_sentence1_data, train_sentence2_data, train_labels_data = load(TRAIN_PREPROCESS_FILE_NAME)
    dev_sentence1_data, dev_sentence2_data, dev_labels_data = load(DEV_PREPROCESS_FILE_NAME)
    word2idx = load(WORD2IDX_FILE_NAME)

    VOCAB_SIZE = len(word2idx) + 1

    # Preparing Dataloader
    print('Preparing DataLoader')
    train_dataset = NliDataset(train_sentence1_data, train_sentence2_data, train_labels_data)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    dev_dataset = NliDataset(dev_sentence1_data, dev_sentence2_data, dev_labels_data)
    dev_dataloader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Defining Model
    model = BiLSTM_Model(vocab_size=VOCAB_SIZE,
                         embedding_dim=EMBEDDING_DIM,
                         hidden1_dim=HIDDEN1_DIM,
                         hidden2_dim=HIDDEN2_DIM,
                         hidden3_dim=HIDDEN3_DIM,
                         classes=CLASSES
                         ).to(DEVICE)

    print(model)
    
    # Defining Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Defining Loss Function
    loss_fn = nn.CrossEntropyLoss()

    # Train the Model
    fit(model, train_dataloader, dev_dataloader, optimizer, loss_fn)


def handle_test_part() :

    # Loading the Preprocessed Data
    print('Loading the Preprocessed Data.')
    test_sentence1_data, test_sentence2_data, test_labels_data = load(TEST_PREPROCESS_FILE_NAME)
    sentence1_data, sentence2_data = load_test_sentences(TEST_FILE_NAME)
    test_labels = test_labels_data.tolist()

    # Preparing Dataloader
    print('Preparing DataLoader')
    test_dataset = NliDataset(test_sentence1_data, test_sentence2_data, test_labels_data)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Loading Model
    model = torch.load(MODEL_FILE_NAME)

    # Predicting Labels
    test_labels_predicted = []

    # Set model to eval mode
    model.eval()

    # Disable gradient computation
    with torch.no_grad() :
        for idx, (batch_premise, batch_hypothesis, batch_labels) in tqdm(enumerate(test_dataloader), total = len(test_dataloader), desc = 'Testing') :
            # Moving data to GPU
            batch_premise = batch_premise.to(DEVICE)
            batch_hypothesis = batch_hypothesis.to(DEVICE)

            out = model(batch_premise, batch_hypothesis)
            predicted_label = torch.argmax(out, dim = 1).cpu().tolist()
            test_labels_predicted += predicted_label

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

    # Loading word2idx dictionary
    word2idx = load(WORD2IDX_FILE_NAME)

    # Loading the model
    model = torch.load(MODEL_FILE_NAME)

    # Vectorizing sentences
    sentence1_processed = process_sentence(sentence1)
    sentence2_processed = process_sentence(sentence2)

    sentence1_vectorised = vectorise_sentences([sentence1_processed], word2idx)
    sentence2_vectorised = vectorise_sentences([sentence2_processed], word2idx)

    # Coverting to Tensor
    sentence1_vectorised = torch.LongTensor(sentence1_vectorised).to(DEVICE)
    sentence2_vectorised = torch.LongTensor(sentence2_vectorised).to(DEVICE)

    # Set model to eval mode
    model.eval()

    # Disable gradient computation
    with torch.no_grad() :
        # Predicting
        out = model(sentence1_vectorised, sentence2_vectorised)
        label_predicted = torch.argmax(out, axis = 1).cpu().tolist()

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
    global WORD2IDX_FILE_NAME, TRAIN_PREPROCESS_FILE_NAME, DEV_PREPROCESS_FILE_NAME, TEST_PREPROCESS_FILE_NAME
    global  MODEL_FILE_NAME, RESULT_FILE_NAME
    
    TRAIN_FILE_NAME = f'../data/{dataset_name}_1.0_train.txt'
    DEV_FILE_NAME = f'../data/{dataset_name}_1.0_dev.txt'
    TEST_FILE_NAME = f'../data/{dataset_name}_1.0_test.txt'

    WORD2IDX_FILE_NAME = f'./PreProcessed_data/word2idx_{dataset_name}.pickle'
    TRAIN_PREPROCESS_FILE_NAME = f'./PreProcessed_data/train_preprocessed_{dataset_name}.pickle'
    DEV_PREPROCESS_FILE_NAME = f'./PreProcessed_data/dev_preprocessed_{dataset_name}.pickle'
    TEST_PREPROCESS_FILE_NAME = f'./PreProcessed_data/test_preprocessed_{dataset_name}.pickle'

    MODEL_FILE_NAME = f'./Models/BiLSTM_Word_Level_Model_{dataset_name}.pth'
    RESULT_FILE_NAME = f'./Results/BiLSTM_Word_Level_Results_{dataset_name}.txt'


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

















