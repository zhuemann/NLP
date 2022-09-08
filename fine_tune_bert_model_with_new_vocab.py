#use hugginface conda env
#this is a language model (LM) -- not clasificaiton. It only fine tunes the head.

from transformers import AutoTokenizer, AutoModelWithLMHead
from transformers import RobertaModel
import os
import pandas as pd
import transformers



#raw biobert weights are in Lymphoma_UW_Retrospective/Models

def run_fine_tune_with_new_vocab(
        model_selection = 0, #0=bio_clinical_bert, 1=bio_bert, 2=bert,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        vocab_file = '',  #leave blank if no vocab added, otherwise the filename, eg, 'vocab25.csv'
        reports_file = 'findings_and_impressions_wo_ds_more_syn.csv'
):

    model_type = ['bio_clinical_bert', 'bio_bert', 'bert', 'roberta']

    if model_type[model_selection] == 'bio_clinical_bert':
        tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        model = AutoModelWithLMHead.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    elif model_type[model_selection] == 'bio_bert':
        tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
        model = AutoModelWithLMHead.from_pretrained("dmis-lab/biobert-v1.1")
    elif model_type[model_selection] == 'bert':
        tokenizer = AutoTokenizer.from_pretrained("/Users/zmh001/Documents/language_models/bert/")
        model = AutoModelWithLMHead.from_pretrained("/Users/zmh001/Documents/language_models/bert/")
    elif model_type[model_selection] == 'roberta':
        tokenizer = AutoTokenizer.from_pretrained("/Users/zmh001/Documents/language_models/roberta_large/")
        model = RobertaModel.from_pretrained("/Users/zmh001/Documents/language_models/roberta_large/")

    #get vocab needed to add
    report_direct = 'Z:/Lymphoma_UW_Retrospective/Reports/'

    #if we want to expand vocab file
    save_name_extension = ''
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
        save_name_extension = '_new_vocab'

    #now train the model

    #first, get the data into correct format -- text blocks.
    text_file = reports_file.replace('.csv', '.txt')

    #make file if it doesn't exist
    if not os.path.exists(os.path.join(report_direct,text_file)):
        df_report = pd.read_csv(os.path.join(report_direct, reports_file))
        with open(os.path.join(report_direct,text_file), 'w') as w:
            for i,row in df_report.iterrows():
                entry = str(row["impression_processed"]).replace('\n', ' ')
                w.write(entry + '\n')



    model_direct = os.path.join('C:/Users/zmh001/Documents/language_models/trained_models',
                                model_type[model_selection] + save_name_extension)

    if not os.path.exists(model_direct): os.mkdir(model_direct)
    tokenizer.save_pretrained(model_direct)

    dataset = transformers.LineByLineTextDataset(
        tokenizer = tokenizer,
        file_path = os.path.join(report_direct,text_file),
        block_size = 16
    )

    data_collator = transformers.DataCollatorForLanguageModeling(
        tokenizer = tokenizer, mlm = True, mlm_probability = 0.15
    )

    training_args = transformers.TrainingArguments(
        output_dir = model_direct,
        overwrite_output_dir = True,
        num_train_epochs = 5,
        per_device_train_batch_size = 16,
        save_steps = 10_000,
        save_total_limit = 3,
    )

    trainer = transformers.Trainer(
        model = model,
        args = training_args,
        data_collator = data_collator,
        train_dataset = dataset,
        # prediction_loss_only = True,
    )

    trainer.train()
    trainer.save_model(model_direct)
#END MAIN FUNCTION


### MAIN ####

# run_fine_tune_with_new_vocab(model_selection = 0, vocab_file = 'vocab25.csv', reports_file = 'findings_and_impressions_wo_ds_more_syn.csv')
# run_fine_tune_with_new_vocab(model_selection = 0, vocab_file = '', reports_file = 'findings_and_impressions_wo_ds_more_syn.csv')
# run_fine_tune_with_new_vocab(model_selection = 1, vocab_file = 'vocab25.csv', reports_file = 'findings_and_impressions_wo_ds_more_syn.csv')
# run_fine_tune_with_new_vocab(model_selection = 1, vocab_file = '', reports_file = 'findings_and_impressions_wo_ds_more_syn.csv')
# run_fine_tune_with_new_vocab(model_selection = 2, vocab_file = 'vocab25.csv', reports_file = 'findings_and_impressions_wo_ds_more_syn.csv')
# run_fine_tune_with_new_vocab(model_selection = 2, vocab_file = '', reports_file = 'findings_and_impressions_wo_ds_more_syn.csv')




#evaluate
# model = transformers.AutoModelWithLMHead.from_pretrained(model_direct)
# tokenizer  = transformers.AutoTokenizer.from_pretrained(model_direct)

# nlp_fill = transformers.pipeline('fill-mask', model = model, tokenizer = tokenizer)
# nlp_fill('There is a inflammatory lesion in the patient''s left lung ' + nlp_fill.tokenizer.mask_token)