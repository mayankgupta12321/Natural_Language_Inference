# Importing Libraries
import pickle
import pandas as pd
import scipy.sparse as sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# File Names
TRAIN_FILE_NAME = None
TEST_FILE_NAME = None
VECTORIZER_FILE_NAME = None
MODEL_FILE_NAME = None
RESULT_FILE_NAME = None


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


# Preprocessing the file.
def preprocess_file(filename):
    print('Preprocessing File.')

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
    dataframe['sentence1'] = dataframe['sentence1'].str.lower()
    dataframe['sentence2'] = dataframe['sentence2'].str.lower()

    #  Converting lables, and sentences to array
    labels_data = dataframe['gold_label'].values
    sentence1_data = dataframe['sentence1'].values
    sentence2_data = dataframe['sentence2'].values

    return labels_data, sentence1_data, sentence2_data


# For Training Part
def handle_train_part():
    train_labels, train_sentence1, train_sentence2 = preprocess_file(TRAIN_FILE_NAME)

    print('Vectorizing Sentences.')
    # Vectorizing sentences
    vectorizer = TfidfVectorizer()
    train_data = train_sentence1 + ' ' + train_sentence2
    vectorizer.fit(train_data)
    save(vectorizer, VECTORIZER_FILE_NAME)
    vectorizer = load(VECTORIZER_FILE_NAME)

    train_sentence1_vectorized = vectorizer.transform(train_sentence1)
    train_sentence2_vectorized = vectorizer.transform(train_sentence2)

    train_features = sparse.hstack(
        (train_sentence1_vectorized, train_sentence2_vectorized))

    print('Training the Model.')
    # Training the model
    model = LogisticRegression(
        random_state=0, max_iter=10000, solver='lbfgs', multi_class='auto')
    model.fit(train_features, train_labels)

    # Saving the model
    save(model, MODEL_FILE_NAME)
    model = load(MODEL_FILE_NAME)

    train_labels_predicted = model.predict(train_features)
    train_accuracy = accuracy_score(train_labels, train_labels_predicted)
    print("-------------------------------------------------------")
    print(f'Train Accuracy : {train_accuracy:1.4f}')
    print("-------------------------------------------------------")


# For Testing Part
def handle_test_part():
    test_labels, test_sentence1, test_sentence2 = preprocess_file(TEST_FILE_NAME)

    # Vectorizing sentences
    vectorizer = load(VECTORIZER_FILE_NAME)

    test_sentence1_vectorized = vectorizer.transform(test_sentence1)
    test_sentence2_vectorized = vectorizer.transform(test_sentence2)

    test_features = sparse.hstack(
        (test_sentence1_vectorized, test_sentence2_vectorized))

    print('Loading the Model.')
    # Loading the model
    model = load(MODEL_FILE_NAME)

    test_labels_predicted = model.predict(test_features)
    
    test_accuracy = accuracy_score(test_labels, test_labels_predicted)

    print(f'Writing Results to File : {RESULT_FILE_NAME}')
    write_results_to_file(test_sentence1, test_sentence2, test_labels, test_labels_predicted, test_accuracy)

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
    sentence1 = input('Enter Sentence1 : ')
    sentence2 = input('Enter Sentence2 : ')

    sentence1 = sentence1.lower()
    sentence2 = sentence2.lower()

    # Vectorizing sentences
    vectorizer = load(VECTORIZER_FILE_NAME)
    sentence1_vectorised = vectorizer.transform([sentence1])
    sentence2_vectorised = vectorizer.transform([sentence2])

    test_features = sparse.hstack((sentence1_vectorised, sentence2_vectorised))

    # Loading the model
    model = load(MODEL_FILE_NAME)

    label_predicted = model.predict(test_features)

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
    global TRAIN_FILE_NAME, TEST_FILE_NAME, VECTORIZER_FILE_NAME
    global MODEL_FILE_NAME, RESULT_FILE_NAME

    TRAIN_FILE_NAME = f'../data/{dataset_name}_1.0_train.txt'
    TEST_FILE_NAME = f'../data/{dataset_name}_1.0_test.txt'

    VECTORIZER_FILE_NAME = f'./Models/Tfidf_vectorizer_{dataset_name}.pickle'

    MODEL_FILE_NAME = f'./Models/Logistic_Regression_Model_{dataset_name}.pickle'
    RESULT_FILE_NAME = f'./Results/Logistic_Regression_Results_{dataset_name}.txt'


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
    print("1. Train Model")
    print("2. Test Model on Test file")
    print("3. Test Model on Manual Sentences")
    ch2 = int(input('Enter you choice : '))

    if ch2 == 1:  # train
        handle_train_part()

    elif ch2 == 2:  # test
        handle_test_part()

    elif ch2 == 3:  # inference
        handle_inference_part()

    else:
        print("Invalid Input.")
        exit()
