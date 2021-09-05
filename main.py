# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import ssl
import pandas as pd
import os
import re
import torch.cuda
import transformers
from transformers import AutoTokenizer, AutoModelWithLMHead
from evaluator import Evaluator
from classification_model_bert_unchanged import fine_tune_model, test_saved_model
import numpy as np

from strip_deauville import run_deauville_stripping
from strip_deauville import highest_deauville
from report_preprocessing import run
from bert_fine_tuning import bert_fine_tuning
from fine_tune_bert_model_with_new_vocab import run_fine_tune_with_new_vocab

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    print("running main: ")
    # data = load_data('Z:\Lymphoma_UW_Retrospective\Reports\indications.xlsx')
    # print(data)
    # run_deauville_stripping()
    # read in full data
    # run_extract_multiple_ds()

    #bert_fine_tuning()

    # model = transformers.AutoModelWithLMHead.from_pretrained('C:/Users/zmh001/Documents/language_models/trained_models/')

    #tokenizer = transformers.AutoTokenizer.from_pretrained('C:/Users/zmh001/Documents/language_models/trained_models/bert_new_vocab/')




    # nlp_fill = transformers.pipeline('fill-mask', model=model, tokenizer=tokenizer)
    #print(nlp_fill('There is a inflammatory lesion in the patient''s left lung ' + nlp_fill.tokenizer.mask_token))
    #print(type(nlp_fill('Dan is a ' + nlp_fill.tokenizer.mask_token)))

    #new_string = 'the lesion in the patient '
    #new_string = 'the meeting on tuesday'
    #for i in range(0,10):

    #    full_return = nlp_fill(new_string + str(nlp_fill.tokenizer.mask_token))
    #    return_dict = full_return[0]
    #    new_string = return_dict['sequence']
    #    print(new_string)



    report_direct = 'Z:/Lymphoma_UW_Retrospective/Reports/'
    report_files = ['ds1_findings_and_impressions_wo_ds_more_syn.csv',
                    'ds2_findings_and_impressions_wo_ds_more_syn.csv',
                    'ds3_findings_and_impressions_wo_ds_more_syn.csv',
                    'ds4_findings_and_impressions_wo_ds_more_syn.csv',
                    'ds5_findings_and_impressions_wo_ds_more_syn.csv']

    #df = pd.DataFrame(columns=['id', 'text'])
    #for i, file in enumerate(report_files):
    #    df0 = pd.read_csv(os.path.join(report_direct, file))
    #    if i <= 2:
    #        df0['labels'] = 0
    #    else:
    #        df0['labels'] = 1
    #    df = pd.concat([df, df0], axis=0, join='outer')

    #print(df)
    #print(df.shape)
    #print(df.size)

    #df.to_csv('Z:/Zach_Analysis/text_data/test_data_binary.csv')
    #df.to_csv('C:/Users/zmh001/Documents/report_data/test_data_binary.csv')

    #run_fine_tune_with_new_vocab()


    Fine_Tune = False
    if Fine_Tune == True:
        fine_tune_model(
            model_selection=3,  # 0=bio_clinical_bert, 1=bio_bert, 2=bert
            num_train_epochs=3,
            test_fract=0.2,
            valid_fract=0.1,
            truncate_left=True,  # truncate the left side of the report/tokens, not the right
            n_nodes=768,  # number of nodes in last classification layer
            vocab_file='',  # leave blank if no vocab added, otherwise the filename, eg, 'vocab25.csv'
            report_files=['ds123_findings_and_impressions_wo_ds_more_syn.csv',
                      'ds45_findings_and_impressions_wo_ds_more_syn.csv']
        )

    Test_Model = True
    if Test_Model == True:
        test_saved_model(
            model_selection=3,  # 0=bio_clinical_bert, 1=bio_bert, 2=bert
            test_fract=0.2,
            truncate_left=True,  # truncate the left side of the report/tokens, not the right
            n_nodes=768,  # number of nodes in last classification layer
            vocab_file='',  # leave blank if no vocab added, otherwise the filename, eg, 'vocab25.csv'
            # model_direct='/home/tjb129/r-fcb-isilon/research/Bradshaw/Lymphoma_UW_Retrospective/Models/classification_models',
            model_direct= 'C:/Users/zmh001/Documents/language_models/trained_models/new_bert_preprpossed_data',
            report_files=['ds123_findings_and_impressions_wo_ds_more_syn.csv',
                          'ds45_findings_and_impressions_wo_ds_more_syn.csv']
        )


    #Fine_Tune = False
    #if (Fine_Tune == True):
    #    fine_tune_model(
    #    model_selection=2,  # 0=bio_clinical_bert, 1=bio_bert, 2=bert
    #    num_train_epochs=3,
    #    test_fract=0.15,
    #    valid_fract=0.10,
    #    truncate_left=True,  # truncate the left side of the report/tokens, not the right
    #    number_of_classes=1,
    #    n_nodes=768,  # number of nodes in last classification layer
    #    vocab_file='',  # leave blank if no vocab added, otherwise the filename, eg, 'vocab25.csv'
    #    report_files=['single_ds_reports.xlsx']
    #    )

    #Test_Model = False
    #if (Test_Model):
    #    test_saved_model(
    #        model_selection=2,  # 0=bio_clinical_bert, 1=bio_bert, 2=bert
    #        test_fract=0.2,
    #        truncate_left=True,  # truncate the left side of the report/tokens, not the right
    #        number_of_classes=1,

    #        n_nodes=768,  # number of nodes in last classification layer
    #        vocab_file='',  # leave blank if no vocab added, otherwise the filename, eg, 'vocab25.csv'
    #        model_direct='C:/Users/zmh001/Documents/language_models/trained_models/',
    #        report_files=['single_ds_reports.xlsx']
    #    )


    print('done')


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
