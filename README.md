# Natural_Language_Inference

Preprocessed Data, Trained Models, and Results are placed [here↗️](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/mayank_gupta_students_iiit_ac_in/EiP2mA6tMMxFkn1hYUZA6dYBm7DtlKlV7Egya8JaDHkoPg)

- Do not change directory structure.
- Before executing below commands make sure you are in home directory of the project.
---------------------------------------------------------------------------------------

### Logistic Regression

- Execute the code using below command :
```
cd Logistic_Regression
python Logistic_Regression.py
```

- It will prompt for asking, which dataset you want to use. `Press 1 for SNLI, 2 for MultiNLI`
- Then It will prompt for asking, whether you want to train model, test model, or inference on input sentences.
`Press 1 for Train Model, 2 for Test Model, 3 for Inference on Input Sentences`
- You can do these operations sequentially. i.e. first train the model, then test the model.
---------------------------------------------------------------------------------------

### BiLSTM Char Level Embedding

- Execute the code using below command :
```
cd BiLSTM_Char_Level_Embedding
python BiLSTM_Char_Level_Model.py
```

- It will prompt for asking, which dataset you want to use. `Press 1 for SNLI, 2 for MultiNLI`
- Then It will prompt for asking, whether you want to Preprocess Data, train model, test model, or inference on input sentences.
`Press 1 for Preprocess Data, 2 for Train Model, 3 for Test Model, 4 for Inference on Input Sentences`
- You can do these operations sequentially. i.e. first preprocess the data, then train the model, then test the model.
---------------------------------------------------------------------------------------

### BiLSTM Word Level Embedding

- Execute the code using below command :
```
cd BiLSTM_Word_Level_Embedding
python BiLSTM_Word_Level_Model
```

- It will prompt for asking, which dataset you want to use. `Press 1 for SNLI, 2 for MultiNLI`
- Then It will prompt for asking, whether you want to Preprocess Data, train model, test model, or inference on input sentences.
`Press 1 for Preprocess Data, 2 for Train Model, 3 for Test Model, 4 for Inference on Input Sentences`
- You can do these operations sequentially. i.e. first preprocess the data, then train the model, then test the model.
---------------------------------------------------------------------------------------

### BERT Model
- Execute the code using below command :

```
cd BERT
python BERT_Model.py
```

- It will prompt for asking, which dataset you want to use. `Press 1 for SNLI, 2 for MultiNLI`
- Then It will prompt for asking, whether you want to Preprocess Data, train model, test model, or inference on input sentences.
`Press 1 for Preprocess Data, 2 for Train Model, 3 for Test Model, 4 for Inference on Input Sentences`
- You can do these operations sequentially. i.e. first preprocess the data, then train the model, then test the model.
---------------------------------------------------------------------------------------

#### Directory Structure
```
Natural_Language_Inference:
|
│   NLP_Project_Report.pdf
│   README.md
│
├───BERT
│   │   BERT_Model.py
│   │
│   ├───Models
│   │       BERT_Model_multinli.pt
│   │       BERT_Model_snli.pt
│   │
│   ├───Preprocessed_data
│   │       dev_preprocessed_multinli.pickle
│   │       dev_preprocessed_snli.pickle
│   │       test_preprocessed_multinli.pickle
│   │       test_preprocessed_snli.pickle
│   │       train_preprocessed_multinli.pickle
│   │       train_preprocessed_snli.pickle
│   │
│   └───Results
│           BERT_Results_multinli.txt
│           BERT_Results_snli.txt
│
├───BiLSTM_Char_Level_Embedding
│   │   BiLSTM_Char_Level_Model.py
│   │
│   ├───Models
│   │       BiLSTM_Char_Level_Model_multinli.h5
│   │       BiLSTM_Char_Level_Model_snli.h5
│   │
│   ├───Preprocessed_data
│   │       char2idx_multinli.pickle
│   │       char2idx_snli.pickle
│   │       char_emeddings_multinli.pickle
│   │       char_emeddings_snli.pickle
│   │       test_preprocessed_multinli.pickle
│   │       test_preprocessed_snli.pickle
│   │       train_preprocessed_multinli.pickle
│   │       train_preprocessed_snli.pickle
│   │
│   └───Results
│           BiLSTM_Char_Level_Results_multinli.txt
│           BiLSTM_Char_Level_Results_snli.txt
│
├───BiLSTM_Word_Level_Embedding
│   │   BiLSTM_Word_Level_Model.py
│   │
│   ├───Models
│   │       BiLSTM_Word_Level_Model_multinli.pth
│   │       BiLSTM_Word_Level_Model_snli.pth
│   │
│   ├───Preprocessed_data
│   │       dev_preprocessed_multinli.pickle
│   │       dev_preprocessed_snli.pickle
│   │       test_preprocessed_multinli.pickle
│   │       test_preprocessed_snli.pickle
│   │       train_preprocessed_multinli.pickle
│   │       train_preprocessed_snli.pickle
│   │       word2idx_multinli.pickle
│   │       word2idx_snli.pickle
│   │
│   └───Results
│           BiLSTM_Word_Level_Results_multinli.txt
│           BiLSTM_Word_Level_Results_snli.txt
│
├───data
│   │   multinli_1.0_dev.txt
│   │   multinli_1.0_test.txt
│   │   multinli_1.0_train.txt
│   │   snli_1.0_dev.txt
│   │   snli_1.0_test.txt
│   │   snli_1.0_train.txt
│   │
│   ├───extras
│   │       multinli_1.0_train_downloaded.txt
│   │       multinli_data_split.ipynb
│   │
│   └───glove.6B
│           glove.6B.300d.txt
│
└───Logistic_Regression
    │   Logistic_Regression.py
    │
    ├───Models
    │       Logistic_Regression_Model_multinli.pickle
    │       Logistic_Regression_Model_snli.pickle
    │       Tfidf_vectorizer_multinli.pickle
    │       Tfidf_vectorizer_snli.pickle
    │
    └───Results
            Logistic_Regression_Results_multinli.txt
            Logistic_Regression_Results_snli.txt
```
       
