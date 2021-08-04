#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 17:01:49 2019

@author: tjb129
"""

import pandas as pd
import os
from nltk.tokenize import word_tokenize, sent_tokenize
import gensim
from report_preprocessing import remove_useless_sentences, replace_section_headers
from nlp_utils import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score
from sklearn import linear_model
import numpy as np

############################
#######    doc2vec    ######
############################

def clean_up_text(df, column_heading='text', save_name = '', remove_useless_sentences=1):
    #do stemming, synonym analysis, clean section headers, etc

    direct = '/home/tjb129/r-fcb-isilon/research/Bradshaw/Lymphoma_UW_Retrospective/Reports'
    synonyms_file = 'synonyms_in_reports.xlsx'
    synonyms_sheet = 'synonyms'

    save_file = 'indications_processed.xlsx'
    save_sheet = 'impression_processed'

    # read in synonyms, defined by user
    syns = pd.read_excel(os.path.join(direct, synonyms_file), synonyms_sheet)
    words_to_replace = syns['Word']
    replacements = syns['Replacement']

    # read in ngram replacements, determined after ngram analysis, words are after
    # stemming
    ngram_file = 'ngram_replacements.xlsx'
    ngram_sheet = 'ngram'
    ngram_syns = pd.read_excel(os.path.join(direct, ngram_file), ngram_sheet)
    ngram_words_to_replace = ngram_syns['Word']
    ngram_replacements = ngram_syns['Replacement']


    # an additional file of medical terms and their synonyms
    CLEVER_file = 'CLEVER_terminology.xlsx'
    CLEVER_sheet= 'clever'
    CLEVER_syns = pd.read_excel(os.path.join(direct, CLEVER_file), CLEVER_sheet)
    CLEVER_to_replace = CLEVER_syns['Word']
    CLEVER_replacements = CLEVER_syns['Replacement']



    findings = df[column_heading]
    filtered_findings = []

    #!!!!!!!!!!!!!!!!!!!!!!!!HOW TO CORRECT SPELLING ERRORS?    #!!!!!!!!!!!!!!!!!!!!!!!!

    # loop through each report
    for i, text in enumerate(findings):
        if i % 100 == 0:
            print(i)    #print out progress every so often

        if type(text) is not str:
            filtered_findings.append('nan')

        else:
            #remove punctuation
            text_filt = clean_text(text)
            #replace with custom synonyms from file
            text_filt = replace_custom_synonyms(text_filt, words_to_replace, replacements)
            #replace numbers with specfici formats
            text_filt = simplify_numbers(text_filt)
            if remove_useless_sentences == 1:
                #remove common but useless sentences
                text_filt = remove_useless_sentences(text_filt)
            #contractions
            text_filt = remove_contractions_and_hyphens(text_filt)
            #replace synonyms from CLEVER vocab list
            text_filt = replace_custom_synonyms(text_filt, CLEVER_to_replace, CLEVER_replacements)
            #remove stope words, break sentences into tokens
            text_filt = remove_stop_words_and_tokenize(text_filt)
    #        text_filt = lemmatize_text(text_filt)
            #stem words
            text_filt = stemming_text(text_filt)
            #here we take tokenized text and untokenize it so we can save it
            text_untok = "".join([" "+j if not j.startswith("'") and j not in string.punctuation else j for j in text_filt]).strip()
            #standardize common section headings
            text_untok = replace_section_headers(text_untok)
            #now do replacements based on ngram analysis
            text_untok = replace_custom_synonyms(text_untok, ngram_words_to_replace, ngram_replacements)
            #output
            filtered_findings.append(text_untok)

    #remove low frequency words
    filtered_findings = remove_low_frequency_words_and_repeated_words(filtered_findings, freq=5)
    #save
    df[column_heading + '_processed'] = filtered_findings
    if not save_name == '':
        df.to_csv(save_name)

    return df





def train_doc2vec_with_ds(
        model_version,
        report_processed = 'doc2vec_ds_5_classes_processed.csv', #processed, cleaned text, containing all cases and labels, optional, otherwise use report_files
        report_files = ['ds1_findings_and_impressions_wo_ds_more_syn.csv', \
                        'ds2_findings_and_impressions_wo_ds_more_syn.csv', \
                        'ds3_findings_and_impressions_wo_ds_more_syn.csv', \
                        'ds4_findings_and_impressions_wo_ds_more_syn.csv', \
                        'ds5_findings_and_impressions_wo_ds_more_syn.csv'],
):
    #trains a doc2vec model, saves it, returns path/name of model
    print('processing data')
    #first load the data files
    direct = '/home/tjb129/r-fcb-isilon/research/Bradshaw/Lymphoma_UW_Retrospective/Reports'
    #load data first
    report_direct = '/home/tjb129/r-fcb-isilon/research/Bradshaw/Lymphoma_UW_Retrospective/Reports'
    if not os.path.exists( os.path.join(report_direct, report_processed) ): #if we need to read the non-processed data, process it
        df = pd.DataFrame(columns=['id','text'])
        for i,file in enumerate(report_files):
            df0 = pd.read_csv(os.path.join(report_direct, file))
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

        #preprocess text data, stemming, synonym, etc
        df = clean_up_text(df, column_heading='text', save_name = '')
        df.to_csv(os.path.join(report_direct, 'doc2vec_ds_' + str(len(report_files)) + '_classes_processed.csv'))
    else:
        df = pd.read_csv(os.path.join(report_direct, report_processed))
    df = df.sort_values('id')
    #get into tagged sentences
    tagged_docs = []
    for i, report_i in enumerate(df['text_processed']):
        if report_i == report_i:
            # for sent_i in sent_tokenize(report_i):
            #     tagged_sentences.append(gensim.models.doc2vec.TaggedDocument(words=word_tokenize(sent_i), tags=[str(i)]))
            tagged_docs.append(gensim.models.doc2vec.TaggedDocument(words=gensim.utils.simple_preprocess(report_i), tags = [str(i)]))


    model_direct = '/home/tjb129/r-fcb-isilon/research/Bradshaw/Lymphoma_UW_Retrospective/Models'
    save_model_name = 'doc2vec_ds_model_' + model_version + '.model'


    if model_version == 'a':
        max_epochs = 100
        vec_size = 200
        window=5
        min_count = 10
        dm = 1
    elif model_version == 'b':
        max_epochs = 100
        vec_size = 300
        window=5
        min_count = 10
        dm = 1
    elif model_version == 'c':  #see https://www.aclweb.org/anthology/W16-1609
        max_epochs = 100
        vec_size = 300
        window=15
        min_count = 10
        dm = 0
    elif model_version == 'd':
        max_epochs = 100
        vec_size = 50
        window=15
        min_count = 10
        dm = 1

    print(f"training doc2vec model {model_version}, {max_epochs} epochs, {vec_size} vec_size, {window} window, {dm} dm")
    model = gensim.models.doc2vec.Doc2Vec(vector_size=vec_size,
                                          window=window,
                                          min_count=min_count,
                                          epochs=max_epochs,
                                          dm = dm)
    model.build_vocab(tagged_docs)
    model.train(tagged_docs, epochs = model.epochs, total_examples = model.corpus_count)

    print('Saving model ' + save_model_name)
    model.save(os.path.join(model_direct, save_model_name))

    return os.path.join(model_direct, save_model_name)



#
#
# def test_a_line(df, model, which_line_to_test = 100):
#
#     test1 = df['impression_processed'][which_line_to_test]
#     #    for sent_i in sent_tokenize(test1):
#     #        test1_tok.append(word_tokenize(sent_i))
#     v1 =model.infer_vector(word_tokenize(test1))
#     similar_doc = model.docvecs.most_similar(positive=[v1], topn=12)
#     print('Original:' + test1 + '\nSimilar docs:')
#     for i, s in enumerate(similar_doc):
#         print(str(i) + ' (' + s[0] +  '): ' + df['impression_processed'][int(s[0])] + ' -- ' + str("{0:.2f}".format(s[1])))
#
#     #dissimilar
#     dissimilar_doc = model.docvecs.most_similar(negative=[v1], topn=12)
#     print('Original:' + test1 + '\nDissimilar docs:')
#     for i, s in enumerate(dissimilar_doc):
#         print(str(i) + ' (' + s[0] +  '): ' + df['impression_processed'][int(s[0])] + ' -- ' + str("{0:.2f}".format(s[1])))
#
#
# def save_vectors(model_version = 'a',
#                  model_path='/home/tjb129/r-fcb-isilon/research/Bradshaw/Lymphoma_UW_Retrospective/Models/doc2vec_model_a.model'):
#     direct = '/home/tjb129/r-fcb-isilon/research/Bradshaw/Lymphoma_UW_Retrospective/Reports'
#     load_file = 'indications_processed.xlsx'
#     load_sheet = 'impression_processed'
#
#     save_file = 'report_vectors.xlsx'
#     save_sheet = 'report_vectors_' + model_version
#
#     model = gensim.models.Doc2Vec.load(model_path)
#
#     # read in full data
#     df = pd.read_excel(os.path.join(direct, load_file), load_sheet)
# #    df_write = pd.DataFrame(columns=['id', 'text', 'vector'])
#     df_write = pd.DataFrame(columns=['id', 'text'])
#     ids = []
#     text = []
# #    vectors = []
#     vector_file = open(os.path.join(direct, save_sheet + '.txt'), "w")
#
#     #get into tagged sentences
#     for i, report_i in enumerate(df['impression_processed']):
#         if report_i == report_i:
#             vector_file.write( " ".join([str(x) for x in model.infer_vector(word_tokenize(report_i))]) + "\n")
#             ids.append(i)
#             text.append(report_i)
# #            vectors.append(v1)
#     vector_file.close()
#     df_write['id'] = ids
#     df_write['text'] = text
# #    df_write['vector'] = vectors
#
#     df_write.to_excel(os.path.join(direct, save_file), sheet_name=save_sheet)
#
#     return os.path.join(direct, save_file), os.path.join(direct, save_sheet + '.txt')
#



def train_classifier_with_doc2vec_model(model_path,
                                        classifier_type = 'logistic',
                                        test_fract = 0.2,
                                        report_processed = 'doc2vec_ds_5_classes_processed.csv', #processed, cleaned text, containing all cases and labels, optional
                                        report_files = ['ds1_findings_and_impressions_wo_ds_more_syn.csv', \
                                                        'ds2_findings_and_impressions_wo_ds_more_syn.csv', \
                                                        'ds3_findings_and_impressions_wo_ds_more_syn.csv', \
                                                        'ds4_findings_and_impressions_wo_ds_more_syn.csv', \
                                                        'ds5_findings_and_impressions_wo_ds_more_syn.csv'],
                                        decoy_report = ''


):
     #get binary labels
    report_direct = '/home/tjb129/r-fcb-isilon/research/Bradshaw/Lymphoma_UW_Retrospective/Reports'
    print('processing classification data')
    if report_processed == '':
        df = pd.DataFrame(columns=['id','text'])
        for i,file in enumerate(report_files):
            df0 = pd.read_csv(os.path.join(report_direct, file))
            df0['labels'] = i
            # df0 = df0.set_index('id')
            df = pd.concat([df, df0], axis=0, join='outer')
            # df = df0.append(df1)
        df = df.set_index('id')

        #preprocess text data, stemming, synonym, etc
        df = clean_up_text(df, column_heading='text', save_name = '')
    else:
        df = pd.read_csv(os.path.join(report_direct,report_processed))
    df = df.sort_values('id')
     #randomly split into train, test, then validations
    df_train, df_test = train_test_split(df, test_size=test_fract, random_state=33)

    #get into tagged sentences. Tag is just a unique ID for each document
    tagged_docs_train = []
    labels_train = []
    for i, entry_i in df_train.iterrows():
        report_i = entry_i['text_processed']
        label_i = entry_i['labels']
        if report_i == report_i:
            tagged_docs_train.append(gensim.models.doc2vec.TaggedDocument(words=gensim.utils.simple_preprocess(report_i), tags=[str(i)]))
            labels_train.append(label_i)
            # for sent_i in sent_tokenize(report_i):
            #     tagged_sentences_train.append(gensim.models.doc2vec.TaggedDocument(words=word_tokenize(sent_i), tags=[str(i)]))


    model = gensim.models.doc2vec.Doc2Vec.load(model_path)

    #train classifier
    X = []
    for i in range(len(tagged_docs_train)):
        X.append(model.infer_vector(tagged_docs_train[i].words))
    train_x = np.asarray(X)
    train_y = np.asarray(labels_train)

    logreg = linear_model.LogisticRegression()
    logreg.fit(train_x,train_y)

    #test classifier
    tagged_docs_test = []
    labels_test = []
    id_test = []
    for i, entry_i in df_test.iterrows():
        report_i = entry_i['text_processed']
        label_i = entry_i['labels']
        id_i = entry_i['id']
        if report_i == report_i:
            tagged_docs_test.append(gensim.models.doc2vec.TaggedDocument(words=gensim.utils.simple_preprocess(report_i), tags=[str(i)]))
            labels_test.append(label_i)
            id_test.append(id_i)
            # for sent_i in sent_tokenize(report_i):


    X = []
    for i in range(len(tagged_docs_test)):
        X.append(model.infer_vector(tagged_docs_test[i].words))
    test_x = np.asarray(X)
    test_y = np.asarray(labels_test)

    predicted = logreg.predict(test_x)

    print(sum(predicted == test_y) / len(test_y))

    df_test = pd.DataFrame({'id':id_test, 'predicted':predicted, 'label':test_y})
    df_test = df_test.sort_values('id')
    df_test = df_test.set_index('id')


    model_type = model_path[model_path.find('model_')+6]  #a,b,c,or d
    num_classes = int(np.max(test_y) + 1)
    save_name = 'prediction_results_doc2vec_model-' + model_type + '_classes-' + str(num_classes) + '.csv'

    df_test.to_csv(os.path.join('/home/tjb129/r-fcb-isilon/research/Bradshaw/Lymphoma_UW_Retrospective/Analysis', 'doc2vec', save_name))


    #if we want to test on our decoy report (sentences replaced with opposite statements)
    if not decoy_report == '':
        df_decoy = pd.read_csv(os.path.join(report_direct, decoy_report))
         #test classifier
        tagged_docs_test = []
        labels_test = []
        id_test = []
        for i, entry_i in df_decoy.iterrows():
            report_i = entry_i['text']
            label_i = entry_i['labels']
            id_i = entry_i['id']
            if report_i == report_i:
                tagged_docs_test.append(gensim.models.doc2vec.TaggedDocument(words=gensim.utils.simple_preprocess(report_i), tags=[str(i)]))
                labels_test.append(label_i)
                id_test.append(id_i)
                # for sent_i in sent_tokenize(report_i):


        X = []
        for i in range(len(tagged_docs_test)):
            X.append(model.infer_vector(tagged_docs_test[i].words))
        test_x = np.asarray(X)
        test_y = np.asarray(labels_test)

        predicted = logreg.predict(test_x)

        print(sum(predicted == test_y) / len(test_y))
        print(confusion_matrix(predicted,test_y))

        df_decoy = pd.DataFrame({'id':id_test, 'prediction':predicted, 'label':test_y})
        df_decoy = df_decoy.sort_values('id')
        df_decoy = df_decoy.set_index('id')


        model_type = model_path[model_path.find('model_')+6]  #a,b,c,or d
        num_classes = int(np.max(test_y) + 1)
        save_name = 'DECOY_prediction_results_doc2vec_model-' + model_type + '_classes-' + str(num_classes) + '.csv'


        df_decoy.to_csv(os.path.join('/home/tjb129/r-fcb-isilon/research/Bradshaw/Lymphoma_UW_Retrospective/Analysis', 'doc2vec', save_name))


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



############### MAIN #############################
def main():
    report_files = ['ds1_findings_and_impressions_wo_ds_more_syn.csv', \
                            'ds2_findings_and_impressions_wo_ds_more_syn.csv', \
                            'ds3_findings_and_impressions_wo_ds_more_syn.csv', \
                            'ds4_findings_and_impressions_wo_ds_more_syn.csv', \
                            'ds5_findings_and_impressions_wo_ds_more_syn.csv']
    # report_files = ['ds123_findings_and_impressions_wo_ds_more_syn.csv',
    #                         'ds45_findings_and_impressions_wo_ds_more_syn.csv']
    report_processed = 'doc2vec_ds_' + str(len(report_files)) + '_classes_processed.csv' #skips processing of the 5 report files
    decoy_report = 'DECOYS_doc2vec_' + str(len(report_files)) + '_classes.csv'  #if we want to evaluate the examples with replaced sentences

    versions = ['a', 'b', 'c', 'd']
    # model_path = '/home/tjb129/r-fcb-isilon/research/Bradshaw/Lymphoma_UW_Retrospective/Models/doc2vec_ds_model_a.model'

    for model_version in versions:
        # model_path = train_doc2vec_with_ds( model_version = model_version,
        #     report_processed = report_processed, #processed, cleaned text, containing all cases and labels, optional, otherwise use report_files
        #     report_files = report_files)
        #
        # # model_path = model_path.replace('model_d', 'model_' + model_version)
        # train_classifier_with_doc2vec_model(model_path=model_path,
        #                                     classifier_type = 'logistic',
        #                                     test_fract = 0.2,
        #                                     report_processed = report_processed,
        #                                     decoy_report = decoy_report)


        analysis_direct =  '/home/tjb129/r-fcb-isilon/research/Bradshaw/Lymphoma_UW_Retrospective/Analysis/doc2vec/'
        filename = 'prediction_results_doc2vec_model-' + model_version + '_classes-' + str(len(report_files)) + '.csv'
        quantify_accuracy_of_csv_file(analysis_direct = analysis_direct,
                                  filename = filename,
                                  pred_col = 'predicted',
                                  true_col = 'label')

