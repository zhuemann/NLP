from nlp_utils import *
import os
import csv
import re
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize, sent_tokenize
from classification_model_doc2vec import clean_up_text


# Need to
# 1. Remove dates
# 2. Round decimals
# 3. Find all instances of deauville/deauvil/etc
# 4. Find all duplicate cases


#def highest_deauville(text):
#    highest = ''
#    scores_found = 0
#    for i in range(1, 6):
#        if text.find('deauvil_score_' + str(i)) > -1:
#            highest = 'deauvil_score_' + str(i)
#            scores_found += 1
#    return highest, scores_found


def highest_deauville(text):
    num_scores = 0
    scores_found = ''
    #index = text.find('deauvil_score_')
    indices = [m.start() for m in re.finditer('deauvil_score_', text)]
    for index in indices:
        for offset in range(1,6):
            for ds in range(1, 6):
                if (index + offset + 13 >= len(text)):
                    continue


                if str(text[index + offset + 13]) == str(ds):

                    if (scores_found.find(str(ds)) > -1):
                        continue

                    scores_found += str(ds)
                    num_scores += 1
                    continue

    return num_scores, scores_found


def remove_deauville(text_filt):
    replacement_word = 'score'
    text_filt = text_filt.replace('deauvil_score_1', replacement_word)
    text_filt = text_filt.replace('deauvil_score_2', replacement_word)
    text_filt = text_filt.replace('deauvil_score_3', replacement_word)
    text_filt = text_filt.replace('deauvil_score_4', replacement_word)
    text_filt = text_filt.replace('deauvil_score_5', replacement_word)
    return text_filt


def run_deauville_stripping():
    # Loads the impressions/fidings, finds which ones have deauville scoers, does some basic text processing, and
    # saves the text and DS in a new file. Note the options in the first few lines

    # comment out the onese you don't want
    options = ['', '']
    options[0] = 'combo'  # combine findings and impression
    # options[0] = 'either' #take impression only, or finding in absece of impression
    options[1] = 'more_synonyms'  # replace more synonyms beyond Deauville
    # options[1] = 'no_synonyms' #don't replace more synonyms

    # direct = '/home/tjb129/r-fcb-isilon/research/Bradshaw/Lymphoma_UW_Retrospective/Reports'
    direct = 'Z:\Lymphoma_UW_Retrospective\Reports'
    indications_file = 'indications.xlsx'

    if options[0] == 'combo':
        mrn_sheet = 'lymphoma_uw_finding_and_impres'
        save_file = 'findings_and_impressions_wo_ds'
    else:
        mrn_sheet = 'lymphoma_uw_finding_or_impres'
        save_file = 'findings_or_impressions_wo_ds'

    if options[1] == 'more_synonyms':
        save_file = save_file + '_more_syn'

    #save_file = save_file + '.csv'
    save_file = save_file + '.xlsx'

    synonyms_file = os.path.join(direct, 'deauville_replacements.xlsx')
    synonyms_sheet = 'ngram'

    more_synonyms_file = os.path.join(direct, 'synonyms_in_reports_v2.xlsx')
    more_synonyms_sheet = 'synonyms'

    # read in full data
    df = pd.read_excel(os.path.join(direct, indications_file), mrn_sheet)

    print(os.path.join(direct, indications_file), mrn_sheet)

    # read in synonyms, defined by user
    syns = pd.read_excel(os.path.join(direct, synonyms_file), synonyms_sheet)
    words_to_replace = syns['Word']
    replacements = syns['Replacement']

    if options[1] == 'more_synonyms':
        more_syns = pd.read_excel(os.path.join(direct, more_synonyms_file), more_synonyms_sheet)
        more_words_to_replace = more_syns['Word']
        more_replacements = more_syns['Replacement']

    findings = df['impression']
    filtered_findings = []
    deauville_scores = []
    hedging = []
    subj_id = []

    # !!!!!!!!!!!!!!!!!!!!!!!!HOW TO CORRECT SPELLING ERRORS?    #!!!!!!!!!!!!!!!!!!!!!!!!

    # loop through each report
    for i, text in enumerate(findings):
        if i % 100 == 0:
            print(i)  # print out progress every so often

        if type(text) is not str:
            filtered_findings.append('nan')
            deauville_scores.append('nan')
            hedging.append(0)

        else:
            # remove punctuation
            text_filt = clean_text(text)
            # remove dates
            text_filt = date_stripper(text_filt)
            # replace with custom synonyms from file-
            text_filt = replace_custom_synonyms(text_filt, words_to_replace, replacements)
            if options[1] == 'more_synonyms':
                text_filt = replace_custom_synonyms(text_filt, more_words_to_replace, more_replacements)
            # replace numbers with specfici formats
            text_filt = simplify_numbers(text_filt)
            # remove common but useless sentences
            # text_filt = remove_useless_sentences(text_filt)
            # # contractions
            text_filt = remove_contractions_and_hyphens(text_filt)
            # get the highest deauville score
            scores_found, ds = highest_deauville(text_filt)
            # remove the deuville scores

            #text_filt = remove_deauville(text_filt) # put this line back in later
            filtered_findings.append(text_filt)
            #deauville_scores.append(ds.replace('deauvil_score_', ''))  # store just the number
            deauville_scores.append(ds)
            hedging.append(scores_found)

    # remove low frequency words
    # filtered_findings = remove_low_frequency_words_and_repeated_words(filtered_findings, freq=10)
    # save



    df['impression_processed'] = filtered_findings
    df['deauville'] = deauville_scores
    df = df.set_index('accession')
    df['num_scores'] = hedging
    # df.to_csv(os.path.join(direct, save_file))

    hedging_report = True

    if hedging_report:
        #save_file = 'multiple_ds_reports' + '.xlsx'
        save_file = 'single_ds_reports' '.xlsx'

        # Save the file out if it is greater than 2
        #df = df[df['num_scores'] >= 2]


        df = df[df['num_scores'] == 1]


    df.to_excel(os.path.join("Z:\\Zach_Analysis\\text_data", save_file))


def run_split_reports_according_to_ds(options='binary'):
    # options can be 'binary' or '5-level'

    direct = '/home/tjb129/r-fcb-isilon/research/Bradshaw/Lymphoma_UW_Retrospective/Reports'
    read_file = 'findings_and_impressions_wo_ds_more_syn.csv'

    df = pd.read_csv(os.path.join(direct, read_file))

    if options == 'binary':
        cut_points = [[1, 2, 3], [4, 5]]
    elif options == '5-level':
        cut_points = [[1], [2], [3], [4], [5]]
    else:
        print('wrong options')
        exit()

    for cut in cut_points:
        category_i = []
        id = []
        for i, row in df.iterrows():
            if row["deauville"] in cut:
                category_i.append(row["impression_processed"])
                id.append(row['accession'])
        df_save = pd.DataFrame(list(zip(id, category_i)), columns=['id', 'text'])
        df_save = df_save.set_index('id')
        save_name = 'ds' + str(cut).replace('[', '').replace(']', '').replace(',', '').replace(' ',
                                                                                               '') + '_' + read_file
        df_save = df_save.sort_values('id')
        df_save.to_csv(os.path.join(direct, save_name))


def interpretability_replace_key_sentences_in_test_dataset(test_fract=0.2,
                                                           test_seed=33,
                                                           num_reports_to_alter=20,
                                                           # report_processed = 'doc2vec_ds_5_classes_processed.csv', #processed, cleaned text, containing all cases and labels, optional
                                                           report_files=[
                                                               'ds123_findings_and_impressions_wo_ds_more_syn.csv',
                                                               'ds45_findings_and_impressions_wo_ds_more_syn.csv'],
                                                           decoy_sentences='replacement_sentences_positive_and_negative.csv'
                                                           ):
    # report_direct = '/home/tjb129/r-fcb-isilon/research/Bradshaw/Lymphoma_UW_Retrospective/Reports'
    # df = pd.read_csv(os.path.join(report_direct,report_processed))
    # df = df.sort_values('id')
    # if not report_processed.find('doc2vec') == -1 :
    #     replacement_sentences = pd.read_csv(os.path.join(report_direct, 'replacement_sentences_positive_and_negative_doc2vec.csv'))
    # else:
    #     replacement_sentences = pd.read_csv(os.path.join(report_direct, 'replacement_sentences_positive_and_negative.csv'))

    # load data first

    report_direct = '/home/tjb129/r-fcb-isilon/research/Bradshaw/Lymphoma_UW_Retrospective/Reports'
    df = pd.DataFrame(columns=['id', 'text'])
    for i, file in enumerate(report_files):
        df0 = pd.read_csv(os.path.join(report_direct, file))
        if len(report_files) > 2:  # multi-class, one-hot
            label_i = [0] * len(report_files)
            label_i[i] = 1
            df0['labels'] = [label_i] * len(df0)
        else:  # binary
            df0['labels'] = i
        # df0 = df0.set_index('id')
        df = pd.concat([df, df0], axis=0, join='outer')
        # df = df0.append(df1)
    df = df.set_index('id')
    df = df.sort_values('id')

    # randomly split into train, test, then validations
    _, df_test = train_test_split(df, test_size=test_fract, random_state=test_seed)
    df_test_replaced = pd.DataFrame(columns=['id', 'labels', 'text_processed'])  # new dataframe with replacements
    df_test = df_test.sort_values('id')
    replacement_sentences = pd.read_csv(os.path.join(report_direct, decoy_sentences))

    replacement_info = []  # store info on which sentences to replace
    for i in range(num_reports_to_alter):
        sentences = sent_tokenize(df_test["text"].iloc[i])
        print('DS = ' + str(df_test["labels"].iloc[i]) + ' , ID = ' + str(df_test.index.values[i]))
        new_report = ''
        for j, sent in enumerate(sentences):
            print(str(j) + ':  ' + sent)
        to_replace = input('Enter id of sentences to replace (eg, 1 13 16 18) : ')
        # to_replace = replacement_info[i][1]
        to_replace = [int(item) for item in to_replace.split()]
        # pos_or_neg = input('Replace with disease-positive sentences (1) or disease-negative (0): ')
        pos_or_neg = 0 if df_test["labels"].iloc[i] > 2 else 1
        replacement_info.append([i, df_test.index.values[i], to_replace])
        head = 'positive_sentence' if pos_or_neg == 1 else 'negative_sentence'
        for j, sent in enumerate(sentences):
            if j in to_replace:
                sent = list(replacement_sentences.sample()[head])
                sent = sent[0]
            new_report = new_report + ' ' + sent
        row_to_add = pd.Series({'labels': df_test["labels"].iloc[i], 'text': new_report, 'id': df_test.index.values[i]})
        df_test_replaced = df_test_replaced.append(row_to_add, ignore_index=True)

        with open('log_file.txt', 'a+') as f:
            f.write(str(i) + "\n")
            f.write(df_test.index.values[i] + '\n')
            f.write(str(to_replace) + "\n")

    df_test_replaced.to_csv(os.path.join(report_direct, 'DECOYS_' + 'bert_' + str(len(report_files)) + '_classes.csv'))
    # now for doc2vec, need special language processing
    df_test_replaced_doc2vec = df_test_replaced.copy()
    df_test_replaced_doc2vec = clean_up_text(df_test_replaced_doc2vec, remove_useless_sentences=0)
    df_test_replaced_doc2vec = df_test_replaced_doc2vec.drop(['text'], axis=1)
    df_test_replaced_doc2vec = df_test_replaced_doc2vec.rename(columns={'text_processed': 'text'})
    df_test_replaced_doc2vec.columns
    df_test_replaced_doc2vec.to_csv(
        os.path.join(report_direct, 'DECOYS_' + 'doc2vec_' + str(len(report_files)) + '_classes.csv'))
