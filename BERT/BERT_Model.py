# Importing Libraries
import pickle
import math
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from transformers import BertTokenizer, BertModel, AdamW
from transformers.optimization import *


# File Names
TRAIN_FILE_NAME = None
DEV_FILE_NAME = None
TEST_FILE_NAME = None

TRAIN_PREPROCESS_FILE_NAME = None
DEV_PREPROCESS_FILE_NAME = None
TEST_PREPROCESS_FILE_NAME = None

MODEL_FILE_NAME = None
RESULT_FILE_NAME = None

MAX_SENT_LEN = 128

# Hyperparameters
BATCH_SIZE = 32
HIDDEN_DIM = 512
CLASSES = 3

LEARNING_RATE = 2e-5
EPS = 1e-6
TRAINING_EPOCHS = 5
WARMUP_PERCENT = 0.2

# Checking if cuda available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on Device : {DEVICE}")
# DEVICE = "cpu"


#  extracting tokenizer from
TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")

# using the tokens from BertTokenizer
SEP_TOKEN = TOKENIZER.sep_token
CLS_TOKEN = TOKENIZER.cls_token
PAD_TOKEN = TOKENIZER.pad_token
UNK_TOKEN = TOKENIZER.unk_token

# using the token ids
SEP_TOKEN_IDX = TOKENIZER.sep_token_id
CLS_TOKEN_IDX = TOKENIZER.cls_token_id
PAD_TOKEN_IDX = TOKENIZER.pad_token_id
UNK_TOKEN_IDX = TOKENIZER.unk_token_id


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
def write_results_to_file(sentence1, sentence2, actual_labels, predicted_labels, test_accuracy):
    # Mapping the numerical representation of tag to corresponding tag
    labels = {
        0: 'entailment',
        1: 'neutral',
        2: 'contradiction'
    }
    with open(RESULT_FILE_NAME, 'w', encoding = 'utf-8') as file:
        file.write(f'Test Accuracy : {test_accuracy:1.4f}\n')
        file.write(
            f'-----------------------------------------------------------\n')
        file.write(
            f'Predicted_Labels || Actual_Labels || Sentence1 || Sentence1\n')
        file.write(
            f'-----------------------------------------------------------\n')
        for sent1, sent2, actual_label, predicted_label in zip(sentence1, sentence2, actual_labels, predicted_labels):
            file.write(
                f'{labels[predicted_label]} || {labels[actual_label]} || {sent1} || {sent2}\n')


# Basic Preprocessing of file.
def preprocess_file(filename):
    print(f'Doing Preprocessing of File {filename}.')

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

    #  Converting dataframe to array
    data_np = dataframe.values

    # defining variables
    k = 0
    processed_data = []

    # looping till all records are processed
    while k < len(data_np):
        temp_state = []
        temp_data = []
        temp_att = []
        temp_token = []
        max_sent_len = 0
        counter = 0

        # Iterating on batch
        for i in range(BATCH_SIZE):
            if k == len(data_np):
                break

            # temp_state
            temp_state.append(labels[data_np[k][0]])

            # temp_data
            sentence = [CLS_TOKEN_IDX]
            sent1 = TOKENIZER.tokenize(data_np[k][1])
            x = TOKENIZER.convert_tokens_to_ids(sent1)
            x.append(SEP_TOKEN_IDX)
            sentence = np.concatenate((sentence, x))

            sent2 = TOKENIZER.tokenize(data_np[k][2])
            x = TOKENIZER.convert_tokens_to_ids(sent2)
            x.append(SEP_TOKEN_IDX)
            sentence = np.concatenate((sentence, x))

            temp_data.append(sentence)

            # temp_att
            att = np.ones(len(sentence), dtype=int)
            temp_att.append(att)

            # temp_token
            x = np.zeros(len(sent1) + 2, dtype=int)
            y = np.ones(len(sent2) + 1, dtype=int)
            token = np.concatenate((x, y))
            temp_token.append(token)

            # Updating max sentence len
            if max_sent_len <= len(sentence):
                max_sent_len = len(sentence)
            
            k += 1

        # Making all data of same length
        for i in range(BATCH_SIZE):
            if i == len(temp_data):
                break

            len_diff = max(0, MAX_SENT_LEN - len(temp_data[i]))

            temp_data[i] = np.concatenate((temp_data[i], np.zeros(len_diff, dtype=int)))[:MAX_SENT_LEN]
            temp_att[i] = np.concatenate((temp_att[i], np.zeros(len_diff, dtype=int)))[:MAX_SENT_LEN]
            temp_token[i] = np.concatenate((temp_token[i], np.ones(len_diff, dtype=int)))[:MAX_SENT_LEN]
                
        processed_data.append([temp_state, temp_data, temp_att, temp_token])

        print(f'Records Processes : {k}/{len(data_np)}\r', end = "")
    print()
    return processed_data


# Loading Test sentences
def load_test_sentences(filename):
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


# Preprocessing part
def handle_preprocess_part():
    # preprocessing the file
    train_data = preprocess_file(TRAIN_FILE_NAME)
    dev_data = preprocess_file(DEV_FILE_NAME)
    test_data = preprocess_file(TEST_FILE_NAME)

    print('Saving the Preprocessed Data.')
    # Saving the Preprocessed Data
    save(train_data, TRAIN_PREPROCESS_FILE_NAME)
    save(dev_data, DEV_PREPROCESS_FILE_NAME)
    save(test_data, TEST_PREPROCESS_FILE_NAME)


# class BERTNLIModel
class BERTNLIModel(nn.Module):
    def __init__(self, bert_model, hidden_dim, classes):
        super().__init__()
        self.bert = bert_model

        embedding_dim = bert_model.config.to_dict()['hidden_size']
        self.out = nn.Linear(embedding_dim, classes)

    # Forward Pass
    def forward(self, sequence, attn_mask, token_type):
        embedded = self.bert(input_ids=sequence,
                             attention_mask=attn_mask,
                             token_type_ids=token_type
                             )[1]

        output = self.out(embedded)
        return output


# function to calculate the accuracy of model
def accuracy(pred, y):
    max_preds = pred.argmax(dim=1, keepdim=True)
    correct = (max_preds.squeeze(1) == y).float()
    return correct.sum() / len(y)


# Training
def train(model, iterator, optimizer, criterion, scheduler):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for batch in tqdm(iterator):
        optimizer.zero_grad()  # clear gradients first
        torch.cuda.empty_cache()  # releases all unoccupied cached memory

        label, sequence, attn_mask, token_type = batch
        label = torch.Tensor(label).to(torch.int64).to(DEVICE)
        sequence = torch.Tensor(sequence).to(torch.int).to(DEVICE)
        attn_mask = torch.Tensor(attn_mask).to(torch.int).to(DEVICE)
        token_type = torch.Tensor(token_type).to(torch.int).to(DEVICE)

        predictions = model(sequence, attn_mask, token_type).to(torch.float32)

        loss = criterion(predictions, label)
        acc = accuracy(predictions, label)

        loss.backward()
        optimizer.step()
        scheduler.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# Testing and Evaluating Loss, Accuracy
def test(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            label, sequence, attn_mask, token_type = batch
            label = torch.Tensor(label).to(torch.int64).to(DEVICE)
            sequence = torch.Tensor(sequence).to(torch.int).to(DEVICE)
            attn_mask = torch.Tensor(attn_mask).to(torch.int).to(DEVICE)
            token_type = torch.Tensor(token_type).to(torch.int).to(DEVICE)

            predictions = model(sequence, attn_mask,
                                token_type).to(torch.float32)

            loss = criterion(predictions, label)
            acc = accuracy(predictions, label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# For Training Part
def handle_train_part():
    # Loading the Preprocessed Data
    print('Loading the Preprocessed Data.')
    train_data = load(TRAIN_PREPROCESS_FILE_NAME)
    dev_data = load(DEV_PREPROCESS_FILE_NAME)

    # Loading BertModel
    print('Loading the Pretrained BertModel.')
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    # Defining Model
    model = BERTNLIModel(bert_model=bert_model,
                         hidden_dim=HIDDEN_DIM,
                         classes=CLASSES).to(DEVICE)

    print('------------------------------------------------------------------------')
    print(model)

    # Defining Optimizer
    optimizer = AdamW(model.parameters(),
                      lr=LEARNING_RATE,
                      eps=EPS,
                      correct_bias=False,
                      no_deprecation_warning=True)

    # Defining Loss Function
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    # Defining Schedular
    total_steps = math.ceil(TRAINING_EPOCHS * len(train_data))
    warmup_steps = int(total_steps * WARMUP_PERCENT)
    scheduler = get_constant_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps)

    # Model Training Started
    print('Model Training Started.')
    best_valid_loss = float('inf')
    for epoch_num in range(TRAINING_EPOCHS):
        print('------------------------------------------------------------------------')
        print(f"Epoch : {epoch_num + 1}")

        # Training
        train_loss, train_acc = train(
            model, train_data, optimizer, criterion, scheduler)
        valid_loss, valid_acc = evaluate(model, dev_data, criterion)

        print(
            f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(
            f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

        # Evaluating
        valid_loss, valid_acc = evaluate(model, dev_state, criterion)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model, MODEL_FILE_NAME)
            print('Best Model Saved.')

        print('------------------------------------------------------------------------')


def handle_test_part():
    # Loading the Preprocessed Data
    print('Loading the Preprocessed Data.')
    test_data = load(TEST_PREPROCESS_FILE_NAME)
    sentence1_data, sentence2_data = load_test_sentences(TEST_FILE_NAME)

    # Loading Model
    model = torch.load(MODEL_FILE_NAME, map_location=DEVICE)

    # Predicting Labels
    test_labels_predicted = []
    test_labels = []

    # Model Evaluation
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_data):
            label, sequence, attn_mask, token_type = batch
            label = torch.Tensor(label).to(torch.int64).to(DEVICE)
            sequence = torch.Tensor(sequence).to(torch.int).to(DEVICE)
            attn_mask = torch.Tensor(attn_mask).to(torch.int).to(DEVICE)
            token_type = torch.Tensor(token_type).to(torch.int).to(DEVICE)

            predictions = model(sequence, attn_mask,
                                token_type).to(torch.float32)
            test_labels_predicted += torch.argmax(
                predictions, dim=1).cpu().tolist()
            test_labels += label.tolist()

    test_accuracy = accuracy_score(test_labels, test_labels_predicted)

    print(f'Writing Results to File : {RESULT_FILE_NAME}')
    write_results_to_file(sentence1_data, sentence2_data,
                          test_labels, test_labels_predicted, test_accuracy)

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


# Process sentences
def process_sentences(sentence1, sentence2):
    # defining variables
    temp_data = []
    temp_att = []
    temp_token = []

    # temp_data
    sentence = [CLS_TOKEN_IDX]
    sent1 = TOKENIZER.tokenize(sentence1)
    x = TOKENIZER.convert_tokens_to_ids(sent1)
    x.append(SEP_TOKEN_IDX)
    sentence = np.concatenate((sentence, x))

    sent2 = TOKENIZER.tokenize(sentence2)
    x = TOKENIZER.convert_tokens_to_ids(sent2)
    x.append(SEP_TOKEN_IDX)
    sentence = np.concatenate((sentence, x))

    temp_data.append(sentence)

    # temp_att
    att = np.ones(len(sentence), dtype=int)
    temp_att.append(att)

    # temp_token
    x = np.zeros(len(sent1) + 2, dtype=int)
    y = np.ones(len(sent2) + 1, dtype=int)
    token = np.concatenate((x, y))
    temp_token.append(token)

    len_diff = max(0, MAX_SENT_LEN - len(temp_data))

    temp_data[0] = np.concatenate((temp_data[0], np.zeros(len_diff, dtype=int)))[:MAX_SENT_LEN]
    temp_att[0] = np.concatenate((temp_att[0], np.zeros(len_diff, dtype=int)))[:MAX_SENT_LEN]
    temp_token[0] = np.concatenate((temp_token[0], np.ones(len_diff, dtype=int)))[:MAX_SENT_LEN]

    return temp_data, temp_att, temp_token


# for inference part from 2 input sentences.
def handle_inference_part():
    sentence1 = input('Enter Sentence1 : ')
    sentence2 = input('Enter Sentence2 : ')

    # Loading the model
    model = torch.load(MODEL_FILE_NAME, map_location=DEVICE)

    # Processing sentences
    sequence, attn_mask, token_type = process_sentences(sentence1, sentence2)

    # Model Evaluation
    model.eval()
    with torch.no_grad():
        sequence = torch.Tensor(sequence).to(torch.int).to(DEVICE)
        attn_mask = torch.Tensor(attn_mask).to(torch.int).to(DEVICE)
        token_type = torch.Tensor(token_type).to(torch.int).to(DEVICE)

        # Prediction
        out = model(sequence, attn_mask, token_type)
        label_predicted = torch.argmax(out, axis=1).cpu().tolist()

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
    global TRAIN_PREPROCESS_FILE_NAME, DEV_PREPROCESS_FILE_NAME, TEST_PREPROCESS_FILE_NAME
    global  MODEL_FILE_NAME, RESULT_FILE_NAME
    global MAX_SENT_LEN
    
    TRAIN_FILE_NAME = f'../data/{dataset_name}_1.0_train.txt'
    DEV_FILE_NAME = f'../data/{dataset_name}_1.0_dev.txt'
    TEST_FILE_NAME = f'../data/{dataset_name}_1.0_test.txt'

    TRAIN_PREPROCESS_FILE_NAME = f'./PreProcessed_data/train_preprocessed_{dataset_name}.pickle'
    DEV_PREPROCESS_FILE_NAME = f'./PreProcessed_data/dev_preprocessed_{dataset_name}.pickle'
    TEST_PREPROCESS_FILE_NAME = f'./PreProcessed_data/test_preprocessed_{dataset_name}.pickle'

    MODEL_FILE_NAME = f'./Models/BERT_Model_{dataset_name}.pt'
    RESULT_FILE_NAME = f'./Results/BERT_Results_{dataset_name}.txt'


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
