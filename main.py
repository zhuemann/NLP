# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import ssl
import pandas as pd
from strip_deauville import run_deauville_stripping

#ssl._create_default_https_context = ssl._create_unverified_context


def load_data(file_path):
    df = pd.read_excel(file_path)
    return df

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('PyCharm')
    #data = load_data('Z:\Lymphoma_UW_Retrospective\Reports\indications.xlsx')
    #print(data)
    run_deauville_stripping()
    print('done')


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
