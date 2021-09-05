import re
import os
import pandas as pd


def check_hedging(text):
    num_scores = 0
    scores_found = ''

    indices = [m.start() for m in re.finditer('deauvil_score_', text)]
    print(indices)
    for index in indices:
        for offset in range(1, 10):
            for ds in range(1, 6):
                if (index + offset + 13 >= len(text)):
                    continue

                if str(text[index + offset + 13]) == str(ds):
                    scores_found += str(ds)
                    num_scores += 1
                    continue

    return num_scores, scores_found

def run_extract_multiple_ds():
    #df = pd.read_excel(os.path.join('Z:\\Lymphoma_UW_Retrospective\\Reports', 'indications_processed.xlsx'),
    #                   'impression_processed')
    #impression = df['impression_processed']

    df = pd.read_excel(os.path.join('Z:\\Lymphoma_UW_Retrospective\\Reports', 'indications.xlsx'),
                                          'lymphoma_uw_finding_and_impres')
    impression = df['impression']

    num_list = []
    scores_found_list = []
    hedged_reports = []
    hedged_reports_file = pd.DataFrame()

    for i, text in enumerate(impression):
        if i % 100 == 0:
            print(i)  # print out progress every so often

        if type(text) is str:

            print(text.find('score'))
            num_ds, ds_found = check_hedging(text)

            num_list.append(num_ds)
            scores_found_list.append(ds_found)

            if ds_found != '' and int(ds_found) > 1:
                hedged_reports.append(text)

    hedges_found = 0
    for i in range(0, len(num_list)):
        if (num_list[i] > 1):
            print(scores_found_list[i])
            hedges_found += 1

    print("hedges_found: ")
    print(hedges_found)
    print('done')

    hedged_reports_file['hedged_reports'] = hedged_reports

    hedged_reports_file.to_excel('Z:\\Zach_Analysis\\text_data\\hedged_reports.xlsx')