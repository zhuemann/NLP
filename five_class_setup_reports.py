from os import listdir
from os.path import isfile, join
from os.path import exists
import pandas as pd
import os


def five_class_setup():

    negative_dir = 'Z:/Lymphoma_UW_Retrospective/Data/mips/Group_1_2_3_curated'
    # negative_dir = '/home/zmh001/r-fcb-isilon/research/Bradshaw/Lymphoma_UW_Retrospective/Data/mips/Group_1_2_3_curated'
    positive_dir = 'Z:/Lymphoma_UW_Retrospective/Data/mips/Group_4_5_curated'
    # positive_dir = '/home/zmh001/r-fcb-isilon/research/Bradshaw/Lymphoma_UW_Retrospective/Data/mips/Group_4_5_curated'

    # gets all the file names in and puts them in a list
    neg_files = [f for f in listdir(negative_dir) if isfile(join(negative_dir, f))]
    pos_files = [f for f in listdir(positive_dir) if isfile(join(positive_dir, f))]

    if "Thumbs.db" in neg_files:
        neg_files.remove("Thumbs.db")
    if "Thumbs.db" in pos_files:
        pos_files.remove("Thumbs.db")

    all_files = neg_files + pos_files

    report_direct = 'Z:\Lymphoma_UW_Retrospective\Reports'
    reports_1 = pd.read_csv(os.path.join(report_direct, 'ds1_findings_and_impressions_wo_ds_more_syn.csv'))
    reports_2 = pd.read_csv(os.path.join(report_direct, 'ds2_findings_and_impressions_wo_ds_more_syn.csv'))
    reports_3 = pd.read_csv(os.path.join(report_direct, 'ds3_findings_and_impressions_wo_ds_more_syn.csv'))
    reports_4 = pd.read_csv(os.path.join(report_direct, 'ds4_findings_and_impressions_wo_ds_more_syn.csv'))
    reports_5 = pd.read_csv(os.path.join(report_direct, 'ds5_findings_and_impressions_wo_ds_more_syn.csv'))

    report_1_labels = pd.DataFrame()
    report_1_labels['labels'] = [0] * len(reports_1)
    labeled_reports =pd.concat([reports_1, report_1_labels], axis=1, join='outer')

    report_2_labels = pd.DataFrame()
    report_2_labels['labels'] = [1] * len(reports_2)
    labeled_report_2 = pd.concat([reports_2, report_2_labels], axis=1, join='outer')
    labeled_reports = pd.concat([labeled_reports, labeled_report_2], axis=0)

    report_3_labels = pd.DataFrame()
    report_3_labels['labels'] = [2] * len(reports_3)
    labeled_report_3 = pd.concat([reports_3, report_3_labels], axis=1, join='outer')
    labeled_reports = pd.concat([labeled_reports, labeled_report_3], axis=0)

    report_4_labels = pd.DataFrame()
    report_4_labels['labels'] = [3] * len(reports_4)
    labeled_report_4 = pd.concat([reports_4, report_4_labels], axis=1, join='outer')
    labeled_reports = pd.concat([labeled_reports, labeled_report_4], axis=0)

    report_5_labels = pd.DataFrame()
    report_5_labels['labels'] = [4] * len(reports_5)
    labeled_report_5 = pd.concat([reports_5, report_5_labels], axis=1, join='outer')
    labeled_reports = pd.concat([labeled_reports, labeled_report_5], axis=0)

    print(labeled_reports)
    return labeled_reports



