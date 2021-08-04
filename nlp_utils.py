import pandas as pd
import os
import re
import string

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import ngrams, FreqDist


def clean_text(text):
    # remove punctuation from the text (leaves periods)
    # Strip HTML tags
    text = date_stripper(text.lower())

    text = re.sub("\n\n", ". ", text)
    text = re.sub("\r\n\r\n", ". ", text)
    text = re.sub("\r\n \r\n", ". ", text)
    text = re.sub(r'[<\[^<\]\+?>@]', ' ', text)
    text = re.sub('\(', ' ', text)
    text = re.sub('=', ' ', text)
    text = re.sub('\)', ' ', text)
    text = re.sub('\r', ' ', text)
    text = re.sub('\n', ' ', text)
    text = re.sub(',', ' ', text)
    text = re.sub('/', ' ', text)
    text = re.sub(':', " : ", text)
    text = re.sub('#', " # ", text)
    text = re.sub('\*', "", text)
    text = re.sub(';', ".", text)
    text = re.sub("\.\.", ".", text)
    text = re.sub(" +", " ", text)  #multiple spaces

    # Strip escaped quotes
    text = text.replace('\\"', '')

    # Strip quotes
    text = text.replace('"', '')

    return text


def date_stripper(text):
    # removes dates in multiple formats, (eg, 1/1/1990 or Jan 1, 1990). Replaces with 'xx'
    expr1 = r"\b\d{2}/\d{2}/[0-9]{2,4}"  # xx/xx/xxxx
    expr2 = r"\b\d{2}-\d{2}-[0-9]{2,4}"  # xx-xx-xxxx
    expr3 = r"\b\d{1}/\d{2}/[0-9]{2,4}"  # x/xx/xxxx
    expr4 = r"\b\d{1}-\d{2}-[0-9]{2,4}"  # x-xx-xxxx
    expr5 = r"\b\d{2}/\d{1}/[0-9]{2,4}"  # xx/x/xxxx
    expr6 = r"\b\d{2}-\d{1}-[0-9]{2,4}"  # xx-x-xxxx
    expr7 = r"\b\d{1}/\d{1}/[0-9]{2,4}"  # x/x/xxxx
    expr8 = r"\b\d{1}-\d{1}-[0-9]{2,4}"  # x-x-xxxx
    expr9 = r"\b\d{4}"  # xxxx

    repl_text = 'datex'

    text = re.sub("|".join([expr1, expr2, expr3, expr4, expr5, expr6, expr7, expr8, expr9]), repl_text, text)

    text = re.sub(
        r'\b(january [0-9]|february [0-9]|march [0-9]|april [0-9]|may [0-9]|june [0-9]|july [0-9]|august [0-9]|september [0-9]|october [0-9]|november [0-9]|december [0-9])\b',
        repl_text, text)
    text = re.sub(
        r'\b(january [0-9]{2}|february [0-9]{2}|march [0-9]{2}|april [0-9]{2}|may [0-9]{2}|june [0-9]{2}|july [0-9]{2}|august [0-9]{2}|september [0-9]{2}|october [0-9]{2}|november [0-9]{2}|december [0-9]{2})\b',
        repl_text, text)
    text = re.sub(
        r'\b(jan [0-9]|feb [0-9]|march [0-9]|april [0-9]|may [0-9]|june [0-9]|july [0-9]|aug [0-9]|sept [0-9]|oct [0-9]|nov [0-9]|dec [0-9])\b',
        repl_text, text)
    text = re.sub(
        r'\b(jan [0-9]{2}|feb [0-9]{2}|march [0-9]{2}|april [0-9]{2}|may [0-9]{2}|june [0-9]{2}|july [0-9]{2}|aug [0-9]{2}|sept [0-9]{2}|oct [0-9]{2}|nov [0-9]{2}|dec [0-9]{2})\b',
        repl_text, text)
    text = re.sub(r'\b(january|february|march|april|june|july|august|september|october|november|december)\b', repl_text,
                  text)
    text = re.sub(r'\b(jan|feb|march|april|june|july|aug|sept|oct|nov|dec)\b', repl_text, text)

    return text


def replace_custom_synonyms(text, words_to_replace, replacements):
    # given a list of 'words_to_replace' and their corresponding 'replacements',
    # goes through and replaces instances of those words. Can handle wildcards
    # in the words_to_replace list (as long as it's at the end). If you want to
    # remove a word, replace it with " in the excel file

    for i in range(0, len(words_to_replace)):
        word = words_to_replace[i]
        #        print(word)
        ind_wild = word.find('*')
        if ind_wild > -1:
            if ind_wild == len(word) - 1:
                word = word[0:ind_wild] + r"\w*"
            else:
                word = word[0:ind_wild] + r'\w*' + word[ind_wild + 1:]

        text = re.sub(r'\b' + word + r'\b', replacements[i], text)
        text = re.sub(r'\"', '', text)  # some words are removed by making them ", remove those here

        # vertebrae
        text = re.sub(r'\b[c][0-9][0-2]?', 'cervical_vertebr', text)
        text = re.sub(r'\b[l][0-9][0-2]?', 'lumbar_vertebr', text)
        text = re.sub(r'\b[t][0-9][0-2]?', 'thoracic_vertebr', text)

    return text


def remove_contractions_and_hyphens(text):
    text = re.sub(r"won\'t", "will not", text)
    text = re.sub(r"can\'t", "can not", text)

    # general
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)

    # replace hyphens with spaces
    text = re.sub(r'\w+(?:-\w+)+', lambda match: match.group(0).replace('-', ' '), text)
    return text


def simplify_numbers(text):
    # Takes numbers from text and simplifies it, either by custom rounding or
    # removal of decimals. This is all specific to PET values (ie, customized)

    # text = 'Hi 0.11, 0.55, 1.44, 1.55, 2.11, 2.66, 3.11, 3.66, 4.22, 4.99, 4, 5.44, 7.5, 8, 9.5, 11, 12.5, 15.5, 21, 29.9, 35.5, 55, 77.7, 99, 104.5, 177, 955'
    text = re.sub(r'([0-9]+)mm', r'\1 mm', text)  # when lazy people forget the space before mm -- use groups!
    text = re.sub(r'([0-9]+)cm', r'\1 cm', text)  # when lazy people forget the space before cm -- use groups!

    text = re.sub(r'\b0\.[0-9]\w*\b', '1', text)  # near 1
    text = re.sub(r'\b1\.[0-4]\w*\b', '1', text)  #
    text = re.sub(r'\b1\.[5-9]\w*\b', '2', text)  #
    text = re.sub(r'\b2\.[0-4]\w*\b', '2', text)  #
    text = re.sub(r'\b2\.[5-9]\w*\b', '3', text)  #
    text = re.sub(r'\b3\.[0-4]\w*\b', '3', text)  #
    text = re.sub(r'\b4\.[0-9]\w*\b', '5', text)  #
    text = re.sub(r'\b[5-7]\b', '5', text)  # Do ones without decimal separately
    text = re.sub(r'\b3\.[5-9]\w*\b', '5', text)  #
    text = re.sub(r'\b[4-6]\.[0-9]*\b', '5', text)  #
    text = re.sub(r'\b[8-9]\b', '10', text)  #
    text = re.sub(r'\b[1-2][0-9]\b', '10', text)  #
    text = re.sub(r'\b[7-9]\.[0-9]*\b', '10', text)  #
    text = re.sub(r'\b[1-2][0-9]\.[0-9]*\b', '10', text)
    text = re.sub(r'\b[3-7][0-9]\b', '50', text)
    text = re.sub(r'\b[3-7][0-9]\.[0-9]*\b', '50', text)
    text = re.sub(r'\b[8-9][0-9]\b', '100', text)
    text = re.sub(r'\b[1-9][0-9][0-9]\b', '100', text)
    text = re.sub(r'\b[7-9][0-9]\.[0-9]*\b', '100', text)
    text = re.sub(r'\b[1-9][0-9][0-9]\.[0-9]*\b', '100', text)

    text = re.sub(r'slice\s[0-9][0-9]*', 'slice', text)  # deal with slice
    text = re.sub(r'[0-9]+[ ]*mm', '1 cm', text)  # data with mm
    text = re.sub(r'([0-9]+\.[0-9]+)(\.)', r'\1 .',
                  text)  # sentence tokenizers have trouble with periods after numbers. Make it easier
    text = re.sub(r'([0-9]+)(\.\s)', r'\1 . ',
                  text)  # sentence tokenizers have trouble with periods after numbers. Make it easier

    return text


def remove_stop_words_and_tokenize(text):
    # remove stop words using nltk. Can cancel out words being deleted by adding
    # them to words_not_excluded. It also tokenizes text
    from nltk.corpus import stopwords
    stopwords = stopwords.words('english')
    words_not_excluded = ["won't", "wouldn't", 'or', 'both', 'no', 'not', 'same']  # is, has, or?
    stopwords_trimmed = list(set(stopwords).difference(set(words_not_excluded)))  # remove words from stopwords list

    text_tok = word_tokenize(text)
    text = []
    for w in text_tok:
        if w not in stopwords_trimmed:
            text.append(w)

    return text


def stemming_text(text):
    # stems text using nltk PorterStemmer
    stemmer = PorterStemmer()
    text_tmp = text
    text = []
    for word in text_tmp:
        text.append(stemmer.stem(word))

    return text


def lemmatize_text(text):
    # doesn't work yet because need part-of-speech info for effective use
    lemmatizer = WordNetLemmatizer()
    text_tmp = text
    text = []
    for word in text_tmp:
        text.append(lemmatizer.lemmatize(word))

    return text


def remove_low_frequency_words_and_repeated_words(dataseries, freq=10):
    # remove low frequency words, those appearing less than 'freq'
    # first, tokenize
    tokenized_words = []
    for text_i in dataseries:
        text_tok = word_tokenize(text_i)
        for word in text_tok:
            tokenized_words.append(word)
    # get a list of most common words (don't want uncommon words)
    freq_analysis = FreqDist(tokenized_words)
    words_to_keep = []
    for key, value in freq_analysis.most_common():
        if value > freq:
            words_to_keep.append(key)
    # now go through each report in the series, filter out  uncommon words
    # report by report, save to filtered_dataseries
    filtered_dataseries = []
    for report_i in dataseries:
        tokenized_report = word_tokenize(report_i)
        filtered_report = []
        word_previous = ''
        for word_i in tokenized_report:
            if word_i in words_to_keep and word_i != word_previous:
                filtered_report.append(word_i)
            word_previous = word_i
        untokenized_filtered_report = "".join(
            [" " + i if not i.startswith("'") and i not in string.punctuation else i for i in filtered_report]).strip()

        filtered_dataseries.append(untokenized_filtered_report)

    return filtered_dataseries

