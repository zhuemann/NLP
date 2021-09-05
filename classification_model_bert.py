# this if for classifying the reports according to deauville scores
#use hugginface conda env

#based on https://colab.research.google.com/github/DhavalTaunk08/Transformers_scripts/blob/master/Transformers_multilabel_distilbert.ipynb#scrollTo=tr6XeiZW3YbT
#Report data comes from strip_deauville.py

from transformers import AutoTokenizer, BertForSequenceClassification, BertModel
import os
import pandas as pd
import transformers
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import numpy as np


import warnings
warnings.simplefilter('ignore')
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import cohen_kappa_score, confusion_matrix, accuracy_score
from sklearn.metrics import hamming_loss
import autograd
from torch.autograd import Variable

import transformers
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import DistilBertTokenizer, DistilBertModel
import logging
logging.basicConfig(level=logging.ERROR)


from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'


def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred)) / \
                    float( len(set_true.union(set_pred)) )
        acc_list.append(tmp_a)
    return np.mean(acc_list)



def truncate_left_text_dataset(dataframe, tokenizer):
    #if we want to only look at the last 512 tokens of a dataset
    #print(dataframe.columns)
    # dataframe = dataframe.drop(columns=['text','labels'])
    # dataframe = dataframe.dropna()
    # dataframe = dataframe.rename(columns={"accession":"id", "impression_processed": "text", "deauville": "labels"})
    for i,row in dataframe.iterrows():
        tokens = tokenizer.tokenize(row['text'])
        # tokens = tokenizer.tokenize(row['impression_processed'])
        strings = tokenizer.convert_tokens_to_string( ( tokens[-512:] ) )
        dataframe.loc[i, 'text'] = strings
        # dataframe.loc[i, 'impression_processed'] = strings

    return dataframe

class MultiLabelDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.targets = self.data.labels
        # self.row_ids = self.data.index
        self.row_ids = self.data.id
        self.max_len = max_len
        # self.batch_size = 1

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        # text = str(self.text[index]) # I changed these since it as index isn't in the df
        text = self.data.iloc[index]
        text = text['text']
        text = " ".join(text.split())
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]
        target = torch.tensor([0, 0, 0, 0, 0], dtype=torch.float)

        # print(self.targets)
        #print(ids)
        #print(text)
        #print(self.targets.iloc[index])
        target = []

        if str(self.targets) == "1.0":
            target = torch.tensor([1, 0, 0, 0, 0], dtype=torch.float)
        if str(self.targets) == "2.0":
            target = torch.tensor([0, 1, 0, 0, 0], dtype=torch.float)
        if str(self.targets) == "3.0":
            target = torch.tensor([0, 0, 1, 0, 0], dtype=torch.float)
        if str(self.targets) == "4.0":
            target = torch.tensor([0, 0, 0, 1, 0], dtype=torch.float)
        if str(self.targets) == "5.0":
            target = torch.tensor([0, 0, 0, 0, 1], dtype=torch.float)


        #print(type(ids))
        #print(ids)
        #print(type(mask))
        #print(mask)

        #print(type(self.row_ids.iloc[index]))
        #print(self.row_ids.iloc[index])
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            # 'targets': torch.tensor(self.targets[index], dtype=torch.float),
            # 'targets': torch.tensor(self.targets.iloc[index], dtype=torch.float),
            # 'targets': torch.tensor(self.targets.iloc[index], dtype=torch.long),
            # 'targets': target,
            'targets' : torch.tensor(self.targets.iloc[index], dtype=torch.long),
            'row_ids': self.row_ids.iloc[index],
        }



class BERTClass(torch.nn.Module):
    def __init__(self, model, n_class, n_nodes):
        super(BERTClass, self).__init__()
        self.l1 = model
        # swapping these layes with kaggle ones
        self.pre_classifier = torch.nn.Linear(n_nodes, n_nodes)
        self.dropout = torch.nn.Dropout(0.3)
        #self.norming_layer = torch.nn.BatchNorm1d(768, affine=False)
        self.layer_norm = torch.nn.LayerNorm(n_nodes)
        #self.relu = torch.nn.LeakyReLU()
        self.classifier = torch.nn.Linear(n_nodes, n_class)
        self.softmax = torch.nn.Softmax()

        self.attention = torch.nn.Sequential(
            torch.nn.Linear(768, 512),
            torch.nn.Tanh(),
            torch.nn.Linear(512, 1),
            torch.nn.Softmax(dim=1)
        )

        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(768, 1)
        )
        #self.linear = torch.nn.Linear(n_nodes, n_class)
        #self.loss = torch.nn.MSELoss()


    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        last_hidden_state = output_1.last_hidden_state
        print("hidden state: ")
        print(last_hidden_state.shape)
        pooler = hidden_state[:, 0]


        #pooler = self.layer_norm(pooler)

        #pooler = self.pre_classifier(pooler)
        #pooler = torch.nn.Tanh()(pooler)

        # pooler = self.norming_layer(pooler)
        # pooler = self.layer_norm(pooler)
        # pooler = self.relu(pooler)
        # pooler = self.dropout(pooler)
        # output = self.classifier(pooler)

        weights = self.attention(last_hidden_state)
        print("weights shape")
        print(weights.shape)
        context_vector = torch.sum(weights * last_hidden_state, dim=1)
        print("context_vector")
        print(context_vector.shape)
        output = self.regressor(context_vector)

        #output = self.softmax(output)
        print("output in forward: ")
        print(output)
        return output

def loss_fn(outputs, targets):
    # outputs = torch.transpose(outputs, 0 ,1)
    print(outputs.shape)
    outputs = torch.squeeze(outputs, dim = 0)
    print(outputs.shape)
    return torch.nn.BCEWithLogitsLoss()(outputs,targets)
    # print(outputs)
    # print(targets)
    # loss = torch.nn.SmoothL1Loss()
    # loss = torch.nn.MSELoss()

    #return loss(outputs, targets)

    #nSamples = [345, 356, 142, 195, 594]
    #normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
    #normedWeights = torch.FloatTensor(normedWeights).to(device)

    #loss = torch.nn.CrossEntropyLoss( weight = normedWeights )
    #targets = targets.long()
    #calc_loss = loss(outputs, targets)
    #return calc_loss

    # loss = torch.nn.MSELoss()
    #loss = one_hot_ce_loss(outputs,targets)
    #return loss

def one_hot_ce_loss(outputs, targets):
    criterion = torch.nn.CrossEntropyLoss()
    _, labels = torch.max(targets, dim = 1)
    return criterion(outputs, labels)

def compute_metrics(pred):
    print("computing metrics")
    print(pred, flush = True)
    print(type(pred))
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    print(precision)
    print(recall)
    print(f1)
    print(pred, flush = True)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def fine_tune_model(
        model_selection = 0, #0=bio_clinical_bert, 1=bio_bert, 2=bert
        num_train_epochs=3,
        test_fract = 0.2,
        valid_fract = 0.1,
        truncate_left = True, #truncate the left side of the report/tokens, not the right
        number_of_classes = 1,
        n_nodes = 768,  #number of nodes in last classification layer
        vocab_file = '',  #leave blank if no vocab added, otherwise the filename, eg, 'vocab25.csv'
        # report_files = ['ds123_findings_and_impressions_wo_ds_more_syn.csv', 'ds45_findings_and_impressions_wo_ds_more_syn.csv' ]
        report_files = ['']
):

    model_type = ['bio_clinical_bert', 'bio_bert', 'bert']
    print(model_type[model_selection])
    if model_type[model_selection] == 'bio_clinical_bert':
        tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", truncation=True)
        model = BertModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    elif model_type[model_selection] == 'bio_bert':
        tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
        model = BertModel.from_pretrained("dmis-lab/biobert-v1.1")
    elif model_type[model_selection] == 'bert':
        #tokenizer = AutoTokenizer.from_pretrained("/Users/zmh001/Documents/language_models/trained_models/")
        model = BertModel.from_pretrained("/Users/zmh001/Documents/language_models/trained_models/")
        tokenizer = AutoTokenizer.from_pretrained("/Users/zmh001/Documents/language_models/bert/")
        #model = BertModel.from_pretrained("/Users/zmh001/Documents/language_models/bert/")


   #if we want to expand vocab file
    # report_direct = '/home/tjb129/r-fcb-isilon/research/Bradshaw/Lymphoma_UW_Retrospective/Reports'
    report_direct = 'Z:/Zach_Analysis/text_data/'
    save_name_extension = '_' + str(len(report_files)) + '_classes'

    if os.path.exists(os.path.join(report_direct, vocab_file)) and not vocab_file == '' :
        vocab = pd.read_csv(os.path.join(report_direct, vocab_file))
        vocab_list = vocab["Vocab"].to_list()

        print(f"Added vocab length: {str(len(vocab_list))}")
        print(f"Original tokenizer length: {str(len(tokenizer))}")

        #add vocab
        tokenizer.add_tokens(vocab_list)

        print(f"New tokenizer length: {str(len(tokenizer))}")

        #expand model
        model.resize_token_embeddings(len(tokenizer))
        save_name_extension = save_name_extension + '_new_vocab'

    # for param in model.parameters():
    #    param.requires_grad = False

    #get binary labels
    # report_direct = '/home/tjb129/r-fcb-isilon/research/Bradshaw/Lymphoma_UW_Retrospective/Reports'
    df = pd.DataFrame(columns=['id','text'])
    for i,file in enumerate(report_files):
        df0 = pd.read_excel(os.path.join(report_direct, file))
        if len(report_files) > 2: #multi-class, one-hot
            label_i = [0] * len(report_files)
            label_i[i] = 1
            df0['labels'] = [label_i] * len(df0)
        else: #binary
            df0['labels'] = i
        # df0 = df0.set_index('id')
        df = pd.concat([df, df0], axis=0, join='outer')
        # df = df0.append(df1)
    df = df.set_index('id')
    df = df.sort_values('id')

    # moved this translation of column names here
    # print(dataframe.columns)
    df = df.drop(columns=['text', 'labels'])
    df = df.dropna()
    # df = df.rename(columns={"accession": "id", "impression_processed": "text", "deauville": "labels"})
    df = df.rename(columns={"accession": "id", "impression_processed": "text", "deauville": "labels"})

    # for i, sample in enumerate(df["labels"]):
    #    if str(df['labels'].iloc[i]) == "1.0":
    #        df['labels'].iloc[i] = [1, 0, 0, 0, 0]
    #    if str(df['labels'].iloc[i]) == "2.0":
    #        df['labels'].iloc[i] = [0, 1, 0, 0, 0]
    #    if str(df['labels'].iloc[i]) == "3.0":
    #        df['labels'].iloc[i] = [0, 0, 1, 0, 0]
    #    if str(df['labels'].iloc[i]) == "4.0":
    #        df['labels'].iloc[i] = [0, 0, 0, 1, 0]
    #    if str(df['labels'].iloc[i]) == "5.0":
    #        df['labels'].iloc[i] = [0, 0, 0, 0, 1]

    df = pd.read_csv('Z:/Zach_Analysis/text_data/test_data_binary.csv')

    if truncate_left:
        save_name_extension = save_name_extension + '_left_trunc'
        df = truncate_left_text_dataset(df,tokenizer)

    print("dataframe")
    print(df)

    # randomly split into train, test, then validations
    df_train, df_test = train_test_split(df, test_size=test_fract, train_size=(1-test_fract), random_state=33)
    df_train, df_valid = train_test_split(df_train, test_size=valid_fract/(1-test_fract), random_state=33)

    training_set = MultiLabelDataset(df_train, tokenizer, 512)
    testing_set = MultiLabelDataset(df_test, tokenizer, 512)

    # print(df_valid)

    valid_set = MultiLabelDataset(df_valid, tokenizer, 512)

    print("FULL Dataset: {}".format(df.shape))
    print("TRAIN Dataset: {}".format(df_train.shape))
    print("TEST Dataset: {}".format(df_test.shape))
    print("VALID Dataset: {}".format(df_valid.shape))

    train_params = {'batch_size': 1, #was 2
                'shuffle': True,
                'num_workers': 0
                }

    test_params = {'batch_size': 1, #was 2
                    'shuffle': True,
                    'num_workers': 0
                    }

    training_loader = DataLoader(training_set, **train_params)
    # testing_loader = DataLoader(testing_set, **test_params)
    valid_loader = DataLoader(valid_set, **test_params)

    #print("before for loop", flush = True)
    #for data in enumerate(training_loader, 0):
    #    print(data)

    #save path
    # model_direct = os.path.join('/home/tjb129/r-fcb-isilon/research/Bradshaw/Lymphoma_UW_Retrospective/Models/classification_models',
    #                            model_type[model_selection] + save_name_extension)
    model_direct = os.path.join( 'C:/Users/zmh001/Documents/language_models/trained_models/new_bert_preprpossed_data/',
                                 model_type[model_selection] + save_name_extension)
    # if not os.path.exists(model_direct): os.mkdir(model_direct)
    # if not os.path.exists(os.path.join(model_direct, 'logs')): os.mkdir(os.path.join(model_direct, 'logs'))

    if len(report_files) > 2:
        n_classes = len(report_files)
    else:
        n_classes = 1

    n_classes = number_of_classes

    # now lets train model
    model_obj = BERTClass(model, n_classes, n_nodes)
    model_obj.to(device)

    optimizer = torch.optim.Adam(params =  model.parameters(), lr=1e-5) # lr was 1e-5


    for epoch in range(num_train_epochs):
        model_obj.train()
        # print(enumerate(training_loader))
        i = 0
        for _, data in tqdm(enumerate(training_loader, 0)):
        # for _, data in enumerate(tqdm(training_loader)):
        # for data in enumerate(training_loader):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)

            text_i = data['row_ids']
            print(i)
            print(text_i)
            print(targets)

            i = i + 1

            outputs = model_obj(ids, mask, token_type_ids)


            optimizer.zero_grad()
            if len(report_files) > 2:
                loss = loss_fn(outputs, targets)
            else:
                #loss = loss_fn(outputs[:,0], targets)
                loss = loss_fn(outputs, targets)
            if _%10==0:
                print(f'Epoch: {epoch}, Loss:  {loss.item()}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #each epoch, look at validation data
        model_obj.eval()
        fin_targets=[]
        fin_outputs=[]
        with torch.no_grad():
            for _, data in tqdm(enumerate(valid_loader, 0)):
                ids = data['ids'].to(device, dtype = torch.long)
                mask = data['mask'].to(device, dtype = torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
                targets = data['targets'].to(device, dtype = torch.float)
                outputs = model_obj(ids, mask, token_type_ids)
                text_i = data['row_ids']
                print(text_i)
                fin_targets.extend(targets.cpu().detach().numpy().tolist())
                # fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
                fin_outputs.extend(outputs.cpu().detach().numpy().tolist())
                print(fin_outputs)

            # final_outputs = np.array(fin_outputs) > 0.5
            # print(final_outputs)

            #final_outputs = np.copy(fin_outputs)
            #final_outputs = (final_outputs == final_outputs.max(axis=1)[:, None]).astype(int)
            # final_outputs = np.array(fin_outputs) > 0.5

            #fin_targets = list(map(int, fin_targets))
            print(fin_targets)

            if number_of_classes > 1:
                final_outputs = np.copy(fin_outputs)

                final_outputs = (final_outputs == final_outputs.max(axis=1)[:, None]).astype(int)
                fin_targets = list(map(int, fin_targets))
                final_outputs_arr = []
                for result in final_outputs:
                    index = np.where(result == 1)[0]
                    final_outputs_arr.append(index[0])
            else:
                final_outputs_arr = fin_outputs

            final_outputs = np.rint(final_outputs_arr)
            print("targets")
            print(fin_targets)
            print("outputs:")
            print(final_outputs)
            val_hamming_loss = metrics.hamming_loss(fin_targets, final_outputs)
            val_hamming_score = hamming_score(np.array(fin_targets), np.array(final_outputs))

            print(f"Epoch {str(epoch)}, Validation Hamming Score = {val_hamming_score}")
            print(f"Epoch {str(epoch)}, Validation Hamming Loss = {val_hamming_loss}")

    torch.save(model_obj.state_dict(), model_direct + 'pretrained_regression_binary_bce')
    # torch.save(model_obj, model_direct + '_full_model')




def test_saved_model(
        model_selection = 0, #0=bio_clinical_bert, 1=bio_bert, 2=bert
        test_fract = 0.2,
        truncate_left = True, #truncate the left side of the report/tokens, not the right
        number_of_classes=1,
        n_nodes = 768,  #number of nodes in last classification layer
        vocab_file = '',  #leave blank if no vocab added, otherwise the filename, eg, 'vocab25.csv'
        model_direct = '/home/tjb129/r-fcb-isilon/research/Bradshaw/Lymphoma_UW_Retrospective/Models/classification_models',
        report_files = ['ds123_findings_and_impressions_wo_ds_more_syn.csv', 'ds45_findings_and_impressions_wo_ds_more_syn.csv' ]
):
    # This funciton is for testing a saved model.

    #first load the model and tokenizer, handle name tags
    #I think you have to load the original model first, just to get the dimensions correct.

    model_type = ['bio_clinical_bert', 'bio_bert', 'bert']
    print(model_type[model_selection])
    if model_type[model_selection] == 'bio_clinical_bert':
        tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", truncation=True)
        model = BertModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    elif model_type[model_selection] == 'bio_bert':
        tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
        model = BertModel.from_pretrained("dmis-lab/biobert-v1.1")
    elif model_type[model_selection] == 'bert':
        tokenizer = AutoTokenizer.from_pretrained("/Users/zmh001/Documents/language_models/bert/")
        #tokenizer = AutoTokenizer.from_pretrained("/Users/zmh001/Documents/language_models/trained_models/")
        model = BertModel.from_pretrained("/Users/zmh001/Documents/language_models/bert/")
        #model = BertModel.from_pretrained("/Users/zmh001/Documents/language_models/trained_models/")



    #load data first
    #report_direct = '/home/tjb129/r-fcb-isilon/research/Bradshaw/Lymphoma_UW_Retrospective/Reports'
    report_direct = 'Z:/Zach_Analysis/text_data/'
    df = pd.DataFrame(columns=['id','text'])
    for i,file in enumerate(report_files):
        df0 = pd.read_excel(os.path.join(report_direct, file))
        if len(report_files) > 2: #multi-class, one-hot
            label_i = [0] * len(report_files)
            label_i[i] = 1
            df0['labels'] = [label_i] * len(df0)
        else: #binary
            df0['labels'] = i
        # df0 = df0.set_index('id')
        df = pd.concat([df, df0], axis=0, join='outer')
        # df = df0.append(df1)
    df = df.set_index('id')
    df = df.sort_values('id')

    # print(dataframe.columns)
    # df = df.read_csv('Z:/Zach_Analysis/text_data/test_data.csv')
    df = df.drop(columns=['text', 'labels'])
    df = df.dropna()
    df = df.rename(columns={"accession": "id", "impression_processed": "text", "deauville": "labels"})

    df = pd.read_csv('Z:/Zach_Analysis/text_data/test_data_binary.csv')

     #randomly split into train, test, then validations. We only need test, delete rest
    _, df_test = train_test_split(df, test_size=test_fract, random_state=33)

    testing_set = MultiLabelDataset(df_test, tokenizer, 512)

    print("TEST Dataset: {}".format(df_test.shape))

    test_params = {'batch_size': 16,
                    'shuffle': True,
                    'num_workers': 0
                    }

    testing_loader = DataLoader(testing_set, **test_params)

      #deal with name tags
    save_name_extension = '_' + str(len(report_files)) + '_classes'
    if not vocab_file == '':
        save_name_extension = save_name_extension + '_new_vocab'
    if truncate_left :
        save_name_extension = save_name_extension + '_left_trunc'
    save_name_extension = save_name_extension + '_state_dict'


    if os.path.exists(os.path.join(report_direct, vocab_file)) and not vocab_file == '' :
        vocab = pd.read_csv(os.path.join(report_direct, vocab_file))
        vocab_list = vocab["Vocab"].to_list()

        print(f"Added vocab length: {str(len(vocab_list))}")
        print(f"Original tokenizer length: {str(len(tokenizer))}")

        #add vocab
        tokenizer.add_tokens(vocab_list)

        print(f"New tokenizer length: {str(len(tokenizer))}")

        #expand model
        model.resize_token_embeddings(len(tokenizer))



    # model_direct = os.path.join('/home/tjb129/r-fcb-isilon/research/Bradshaw/Lymphoma_UW_Retrospective/Models/classification_models',
    #                            model_type[model_selection] + save_name_extension)

    # model_direct = os.path.join('C:/Users/zmh001/Documents/language_models/trained_models/',
    #    model_type[model_selection] + save_name_extension)
    # model_direct = 'C:/Users/zmh001/Documents/language_models/trained_models/pytorch_model.bin'
    model_direct = 'C:/Users/zmh001/Documents/language_models/trained_models/new_bert_preprpossed_data/bert_1_classes_left_truncpretrained_regression_binary_bce'
    # model_direct = 'C:/Users/zmh001/Documents/language_models/trained_models/new_bert_preprpossed_data/bert_1_classes_left_truncpretrained_double_fc_softmax_with_norming_lr_5'

    if len(report_files) > 2:
        n_classes = len(report_files)
    else:
        n_classes = 1

    # n_classes = 5
    n_classes = number_of_classes
    #now lets load the model, then load state
    model_obj = BERTClass(model, n_classes, n_nodes)
    model_obj.load_state_dict(torch.load(model_direct))
    model_obj.to(device)

    #run through test data
    model_obj.eval()
    fin_targets=[]
    fin_outputs=[]
    row_ids = []
    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader, 0)):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            print(ids)
            outputs = model_obj(ids, mask, token_type_ids)
            row_ids.extend(data['row_ids'])
            print(row_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            # fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
            fin_outputs.extend(outputs.cpu().detach().numpy().tolist())

        #get the final score
        if len(report_files) > 2:
            final_outputs = np.copy(fin_outputs)
            final_outputs = (final_outputs == final_outputs.max(axis=1)[:,None]).astype(int)
        else:
            final_outputs = np.copy(fin_outputs)
            final_outputs = (final_outputs == final_outputs.max(axis=1)[:, None]).astype(int)
            #final_outputs = np.array(fin_outputs) > 0.5



        test_hamming_score = hamming_score(np.array(fin_targets), np.array(final_outputs))
        fin_targets = list(map(int, fin_targets))


        if number_of_classes > 1:
            final_outputs = np.copy(fin_outputs)

            final_outputs = (final_outputs == final_outputs.max(axis=1)[:, None]).astype(int)
            # fin_targets = list(map(int, fin_targets))
            final_outputs_arr = []
            for result in final_outputs:
                index = np.where(result == 1)[0]
                final_outputs_arr.append(index[0])
        else:
            final_outputs_arr = fin_outputs

        final_outputs = final_outputs_arr
        final_outputs = np.rint(final_outputs)
        print(final_outputs)

        accuracy = accuracy_score(np.array(fin_targets), np.array(final_outputs))
        print(f"Test Hamming Score = {test_hamming_score}\nTest Accuracy = {accuracy}\n{model_type[model_selection] + save_name_extension}")
        #create a dataframe of the prediction, labels, and which ones are correct
        if len(report_files) > 2:
            df_test_vals = pd.DataFrame(list(zip(row_ids, np.argmax(fin_targets, axis=1).astype(int).tolist(), np.argmax(final_outputs, axis=1).astype(int).tolist())), columns=['id', 'label', 'prediction'])
        else:
            # df_test_vals = pd.DataFrame(list(zip(row_ids, list(map(int, fin_targets)), final_outputs[:,0].astype(int).tolist())), columns=['id', 'label', 'prediction'])
            df_test_vals = pd.DataFrame( list(zip(row_ids, list(map(int, fin_targets)), list(map(int, final_outputs)) )), columns=['id', 'label', 'prediction'])
            # df_test_vals['correct'] = df_test_vals['label'].equals(df_test_vals['prediction'])
        df_test_vals['correct'] = np.where( df_test_vals['label'] == df_test_vals['prediction'], 1, 0)
        df_test_vals = df_test_vals.sort_values('id')
        df_test_vals = df_test_vals.set_index('id')

    return test_hamming_score, df_test_vals






def test_decoy_model(df_decoy,
        nclasses=2,
        model_selection = 0, #0=bio_clinical_bert, 1=bio_bert, 2=bert
        test_fract = 0.2,
        truncate_left = True, #truncate the left side of the report/tokens, not the right
        n_nodes = 768,  #number of nodes in last classification layer
        vocab_file = '',  #leave blank if no vocab added, otherwise the filename, eg, 'vocab25.csv'
        model_direct = '/home/tjb129/r-fcb-isilon/research/Bradshaw/Lymphoma_UW_Retrospective/Models/classification_models'
                     ):
    #for testing the model with data that has been changed (positive sentences swapped with negative sentences)


    model_type = ['bio_clinical_bert', 'bio_bert', 'bert']
    print(model_type[model_selection])
    if model_type[model_selection] == 'bio_clinical_bert':
        tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", truncation=True)
        model = BertModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    elif model_type[model_selection] == 'bio_bert':
        tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
        model = BertModel.from_pretrained("dmis-lab/biobert-v1.1")
    elif model_type[model_selection] == 'bert':
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased")

    testing_set = MultiLabelDataset(df_decoy, tokenizer, 512)

    print("DECOY Dataset: {}".format(df_decoy.shape))

    test_params = {'batch_size': 4,
                    'shuffle': True,
                    'num_workers': 0
                    }

    testing_loader = DataLoader(testing_set, **test_params)

      #deal with name tags
    report_direct = '/home/tjb129/r-fcb-isilon/research/Bradshaw/Lymphoma_UW_Retrospective/Reports'
    save_name_extension = '_' + str(nclasses) + '_classes'
    if  not vocab_file == '':
        save_name_extension = save_name_extension + '_new_vocab'
    if truncate_left :
        save_name_extension = save_name_extension + '_left_trunc'
    save_name_extension = save_name_extension + '_state_dict'


    if os.path.exists(os.path.join(report_direct, vocab_file)) and not vocab_file == '' :
        vocab = pd.read_csv(os.path.join(report_direct, vocab_file))
        vocab_list = vocab["Vocab"].to_list()

        print(f"Added vocab length: {str(len(vocab_list))}")
        print(f"Original tokenizer length: {str(len(tokenizer))}")

        #add vocab
        tokenizer.add_tokens(vocab_list)

        print(f"New tokenizer length: {str(len(tokenizer))}")

        #expand model
        model.resize_token_embeddings(len(tokenizer))



    model_direct = os.path.join('/home/tjb129/r-fcb-isilon/research/Bradshaw/Lymphoma_UW_Retrospective/Models/classification_models',
                                model_type[model_selection] + save_name_extension)

    if nclasses > 2:
        n_classes = nclasses
    else:
        n_classes = 1

    #now lets load the model, then load state
    model_obj = BERTClass(model, n_classes, n_nodes)
    model_obj.load_state_dict(torch.load(model_direct))
    model_obj.to(device)

    #run through test data
    model_obj.eval()
    fin_targets=[]
    fin_outputs=[]
    row_ids = []
    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader, 0)):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model_obj(ids, mask, token_type_ids)
            row_ids.extend(data['row_ids'])
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
        #get the final score
        if nclasses > 2:
            final_outputs = np.copy(fin_outputs)
            final_outputs = (final_outputs == final_outputs.max(axis=1)[:,None]).astype(int)
        else:
            final_outputs = np.array(fin_outputs) > 0.5
        test_hamming_score = hamming_score(np.array(fin_targets), np.array(final_outputs))
        print(f"Test Hamming Score = {test_hamming_score}, {model_type[model_selection] + save_name_extension}")
        #create a dataframe of the prediction, labels, and which ones are correct
        if nclasses > 2:
            df_decoy_vals = pd.DataFrame(list(zip(row_ids, np.argmax(fin_targets, axis=1).astype(int).tolist(), np.argmax(final_outputs, axis=1).astype(int).tolist())), columns=['id', 'label', 'prediction'])
        else:
            df_decoy_vals = pd.DataFrame(list(zip(row_ids, list(map(int, fin_targets)), final_outputs[:,0].astype(int).tolist())), columns=['id', 'label', 'prediction'])
            # df_decoy_vals['correct'] = df_decoy_vals['label'].equals(df_decoy_vals['prediction'])
        df_decoy_vals['correct'] = np.where( df_decoy_vals['label'] == df_decoy_vals['prediction'], 1, 0)
        df_decoy_vals = df_decoy_vals.sort_values('id')
        df_decoy_vals = df_decoy_vals.set_index('id')

    return test_hamming_score, df_decoy_vals


def quantify_accuracy_of_csv_file(analysis_direct = '/home/tjb129/r-fcb-isilon/research/Bradshaw/Lymphoma_UW_Retrospective/Analysis/bert_classification_results',
                                  filename = '',
                                  pred_col = 'prediction',
                                  true_col = 'label'
                                  ):

    df = pd.read_csv(os.path.join(analysis_direct, filename))
    pred = np.asarray(df[pred_col])
    tru = np.asarray(df[true_col])
    kappa = cohen_kappa_score(pred, tru)
    weighted_kappa = cohen_kappa_score(pred, tru, weights='linear')
    conf_mat = confusion_matrix(pred, tru)
    accuracy = sum(pred == tru) / len(tru)
    print(filename)
    print('kappa: ' + str(kappa))
    print('weighted_kappa: ' + str(weighted_kappa))
    print('accuracy: ' + str(accuracy))
    print('confusion_matrix: \n' + str(conf_mat))

##########MAIN##############
def main():
    trunc_options = [False, True]
    vocab_options = ['vocab25.csv', '']
    report_files = ['ds1_findings_and_impressions_wo_ds_more_syn.csv', 'ds2_findings_and_impressions_wo_ds_more_syn.csv', 'ds3_findings_and_impressions_wo_ds_more_syn.csv', 'ds4_findings_and_impressions_wo_ds_more_syn.csv', 'ds5_findings_and_impressions_wo_ds_more_syn.csv']
    # report_files = ['ds123_findings_and_impressions_wo_ds_more_syn.csv', 'ds45_findings_and_impressions_wo_ds_more_syn.csv' ]

    for i in range(3):
        for trunc in trunc_options:
            for vocab in vocab_options:
                ## first train
                # fine_tune_model(model_selection=i, num_train_epochs=5, truncate_left=trunc, vocab_file=vocab, report_files = report_files)

                ##now test
                # score, df_test_vals = test_saved_model(model_selection=i, truncate_left=trunc, vocab_file=vocab, report_files = report_files)

                vocab_tag = 'True' if vocab == '' else 'False'
                save_direct = '/home/tjb129/r-fcb-isilon/research/Bradshaw/Lymphoma_UW_Retrospective/Analysis/bert_classification_results/'
                save_name = 'ds_class_results' + \
                    '_modeltype-' + str(i) + '_classes-' + str(len(report_files)) + '_trunc-' + str(trunc) + '_vocab-' + vocab_tag + '.csv'
                # df_test_vals.to_csv(os.path.join(save_direct,save_name))
                quantify_accuracy_of_csv_file(analysis_direct=save_direct, filename=save_name)



    df_decoy = pd.read_csv('/home/tjb129/r-fcb-isilon/research/Bradshaw/Lymphoma_UW_Retrospective/Reports/DECOYS_bert_5_classes.csv')
    df_decoy = df_decoy.set_index('id')
    hamming, df_decoy_vals = test_decoy_model(df_decoy,
            nclasses=5,
            model_selection = 1, #0=bio_clinical_bert, 1=bio_bert, 2=bert
            test_fract = 0.2,
            truncate_left = False, #truncate the left side of the report/tokens, not the right
            n_nodes = 768,  #number of nodes in last classification layer
            vocab_file = 'vocab25.csv',
             model_direct = '/home/tjb129/r-fcb-isilon/research/Bradshaw/Lymphoma_UW_Retrospective/Models/classification_models')
    df_decoy_vals.to_csv(os.path.join(save_direct, 'DECOY_RESULTS.csv'))
