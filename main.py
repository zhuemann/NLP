# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
import pandas as pd


def load_data(file_path):
    df = pd.read_excel(file_path)
    return df

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('PyCharm')
    data = load_data('Z:\Lymphoma_UW_Retrospective\Reports\indications.xlsx')
    print(data)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
