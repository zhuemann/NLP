#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 12:33:04 2019

Read in the findings/indications, and do text clean up (eg, synonym analysis, 
changing date formats, etc).

@author: tjb129
"""

import pandas as pd
import os
import re
import string
from nlp_utils import *

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import ngrams, FreqDist



def remove_useless_sentences(text):
    #remove sentences that don't have information. Can remove them if they have
    #certain combinations of words. This is custom defined based on commonly
    #used phrases
    output_sentences = []    
    #break into sentences, loop through
    sentences = sent_tokenize(text)    
    for sentence in sentences:
        include = 1
        #remove sentences with normal_biodistirbtion in the brain, extraocular, etc.
        ind1 = re.findall(r'normal_biodistribution', sentence)
        ind2 = re.findall(r'brain', sentence)
        ind3 = re.findall(r'extraocular', sentence)
        ind4 = re.findall(r'salivary', sentence)
        ind5 = re.findall(r'collect', sentence)
        ind6 = re.findall(r'bowel', sentence)
        summed1 = min(1,len(ind1)) + min(1,len(ind2)) + min(1,len(ind3)) + min(1,len(ind4))+ min(1,len(ind5))+ min(1,len(ind6))                          
        if summed1 > 3:
            include = 0     # don't keep this sentence
        
        #remove sentences with extreted tracer present .
        ind1 = re.findall(r'urinar', sentence)
        ind2 = re.findall(r'collect', sentence)
        ind3 = re.findall(r'bladder', sentence)
        ind4 = re.findall(r'excret', sentence)
        ind5 = re.findall(r'normal_biodistribution', sentence)
        ind6 = re.findall(r'system', sentence)
        summed2 = min(1,len(ind1)) + min(1,len(ind2)) + min(1,len(ind3)) + min(1,len(ind4))    + min(1,len(ind5))      + min(1,len(ind6))                  
        if summed2 > 3:
            include = 0
        
        #remove sentences with myocardium.
        ind1 = re.findall(r'normal_biodistribution', sentence)
        ind2 = re.findall(r'myocardium', sentence)
        summed3 = min(1,len(ind1)) + min(1,len(ind2))                          
        if summed3 > 1:
            include = 0
            
        #remove sentences with brain .
        ind1 = re.findall(r'symmetr', sentence)
        ind2 = re.findall(r'hypermetab', sentence)
        ind3 = re.findall(r'brain', sentence)
        summed4 = min(1,len(ind1)) + min(1,len(ind2)) + min(1,len(ind3))                      
        if summed4 > 2:
            include = 0   
            
          #remove sentences with liver .
        ind1 = re.findall(r'heterogen', sentence)
        ind2 = re.findall(r'hypermetabol', sentence)
        ind3 = re.findall(r'liver', sentence)
        ind4 = re.findall(r'spleen', sentence)
        summed5 = min(1,len(ind1)) + min(1,len(ind2)) + min(1,len(ind3)) + min(1,len(ind4))                      
        if summed5 > 3:
            include = 0       
            
         #remove sentences with adrenal .
        ind1 = re.findall(r'adren', sentence)
        ind2 = re.findall(r'gland', sentence)
        ind3 = re.findall(r'unremark', sentence)
        summed6 = min(1,len(ind1)) + min(1,len(ind2)) + min(1,len(ind3))                     
        if summed6 > 2:
            include = 0   
            
         #remove sentences with bowel .
        ind1 = re.findall(r'uptake', sentence)
        ind2 = re.findall(r'bowel', sentence)
        ind3 = re.findall(r'normal_biodistribut', sentence)
        ind4 = re.findall(r'unremark', sentence)
        summed7 = min(1,len(ind1)) + min(1,len(ind2)) + min(1,len(ind3))                     
        if summed7 > 2:
            include = 0   
         
         #remove sentences with exremeties .
        ind1 = re.findall(r'area', sentence)
        ind2 = re.findall(r'abnorm', sentence)
        ind3 = re.findall(r'extrem', sentence)
        ind4 = re.findall(r'no ', sentence)
        summed8 = min(1,len(ind1)) + min(1,len(ind2)) + min(1,len(ind3))    + min(1,len(ind4))                  
        if summed8 > 3:
            include = 0      
        
         #remove sentences with skel .
        ind1 = re.findall(r'hypermet', sentence)
        ind2 = re.findall(r'mass', sentence)
        ind3 = re.findall(r'no ', sentence)
        ind4 = re.findall(r'skel', sentence)
        summed9 = min(1,len(ind1)) + min(1,len(ind2)) + min(1,len(ind3))    + min(1,len(ind4))                  
        if summed9 > 3:
            include = 0
            
         #remove sentences with chest .
        ind1 = re.findall(r'hypermet', sentence)
        ind2 = re.findall(r'identifi', sentence)
        ind3 = re.findall(r'no ', sentence)
        ind4 = re.findall(r'chest', sentence)
        summed10 = min(1,len(ind1)) + min(1,len(ind2)) + min(1,len(ind3))    + min(1,len(ind4))                  
        if summed10 > 3:
            include = 0    
            
        #remove sentences with hila .
        ind1 = re.findall(r'hypermet', sentence)
        ind2 = re.findall(r'axilla', sentence)
        ind3 = re.findall(r'no ', sentence)
        ind4 = re.findall(r'hila', sentence)
        summed11 = min(1,len(ind1)) + min(1,len(ind2)) + min(1,len(ind3))    + min(1,len(ind4))                  
        if summed11 > 3:
            include = 0       
            
        #remove sentences with blood pool .
        ind1 = re.findall(r'mediatstin', sentence)
        ind2 = re.findall(r'blood', sentence)
        ind3 = re.findall(r'suvmax', sentence)
        ind4 = re.findall(r'measur', sentence)
        ind5 = re.findall(r'aorta', sentence)
        ind6 = re.findall(r'thorac', sentence)
        summed12 = min(1,len(ind1)) + min(1,len(ind2)) + min(1,len(ind3))    + min(1,len(ind4))  + min(1,len(ind5))  + min(1,len(ind6))                  
        if summed12 > 3:
            include = 0   
            
        #remove sentences with blood pool .
        ind1 = re.findall(r'physiolog', sentence)
        ind2 = re.findall(r'distribu', sentence)
        ind3 = re.findall(r'abdomen', sentence)
        ind4 = re.findall(r'pelvi', sentence)
        ind5 = re.findall(r'liver', sentence)
        summed13 = min(1,len(ind1)) + min(1,len(ind2)) + min(1,len(ind3))    + min(1,len(ind4))  + min(1,len(ind5))                  
        if summed13 > 3:
            include = 0      
        
        #remove sentences with mastoid
        ind1 = re.findall(r'tympan', sentence)
        ind2 = re.findall(r'mastoid', sentence)
        ind3 = re.findall(r'clear', sentence)
        summed14 = min(1,len(ind1)) + min(1,len(ind2)) + min(1,len(ind3))                    
        if summed14 > 2:
            include = 0 
        
         #remove sentences with sinus .
        ind1 = re.findall(r'paranas', sentence)
        ind2 = re.findall(r'sinus', sentence)
        ind3 = re.findall(r'aerat', sentence)
        summed15 = min(1,len(ind1)) + min(1,len(ind2)) + min(1,len(ind3))                    
        if summed15 > 2:
            include = 0 

         #remove sentences with larynx .
        ind1 = re.findall(r'cavity', sentence)
        ind2 = re.findall(r'orophary', sentence)
        ind3 = re.findall(r'nasophar', sentence)
        ind4 = re.findall(r'unremark', sentence)
        summed16 = min(1,len(ind1)) + min(1,len(ind2)) + min(1,len(ind3))  + min(1,len(ind4))                     
        if summed16 > 3:
            include = 0 
        
         #remove sentences with pulmonary .
        ind1 = re.findall(r'no ', sentence)
        ind2 = re.findall(r'pumonari', sentence)
        ind3 = re.findall(r'mass', sentence)
        ind4 = re.findall(r'identifi', sentence)
        summed17 = min(1,len(ind1)) + min(1,len(ind2)) + min(1,len(ind3))  + min(1,len(ind4))                     
        if summed17 > 3:
            include = 0 

       #remove sentences with pleural .
        ind1 = re.findall(r'no ', sentence)
        ind2 = re.findall(r'pleural', sentence)
        ind3 = re.findall(r'pericardi', sentence)
        summed18 = min(1,len(ind1)) + min(1,len(ind2)) + min(1,len(ind3))                    
        if summed18 > 2:
            include = 0 
            
            
            #remove sentences with bowel .
        ind1 = re.findall(r'bowel', sentence)
        ind2 = re.findall(r'colon', sentence)
        ind3 = re.findall(r'normal', sentence)
        ind4 = re.findall(r'calib', sentence)
        summed19 = min(1,len(ind1)) + min(1,len(ind2)) + min(1,len(ind3)) + min(1,len(ind4))                    
        if summed19 > 3:
            include = 0 
        
        
            #remove sentences with bowel .
        ind1 = re.findall(r'incident', sentence)
        ind2 = re.findall(r'find', sentence)
        ind3 = re.findall(r'attenu', sentence)
        ind4 = re.findall(r'ct ', sentence)
        summed20 = min(1,len(ind1)) + min(1,len(ind2)) + min(1,len(ind3)) + min(1,len(ind4)) 
        if summed20 > 2:
            include = 0 
        
           #remove sentences with pelvis .
        ind1 = re.findall(r'no ', sentence)
        ind2 = re.findall(r'node', sentence)
        ind3 = re.findall(r'mesenter', sentence)
        ind4 = re.findall(r'mass', sentence)
        summed21 = min(1,len(ind1)) + min(1,len(ind2)) + min(1,len(ind3)) + min(1,len(ind4))                    
        if summed21 > 3:
            include = 0 
            
           #remove sentences with aorta .
        ind1 = re.findall(r'thorac', sentence)
        ind2 = re.findall(r'aorta', sentence)
        ind3 = re.findall(r'normal', sentence)
        ind4 = re.findall(r'cours', sentence)
        summed22 = min(1,len(ind1)) + min(1,len(ind2)) + min(1,len(ind3)) + min(1,len(ind4))                    
        if summed22 > 3:
            include = 0   
            
           #remove sentences with ? .
        ind1 = re.findall(r'abdomen', sentence)
        ind2 = re.findall(r'aorta', sentence)
        ind3 = re.findall(r'iliac', sentence)
        ind4 = re.findall(r'normal', sentence)
        summed23 = min(1,len(ind1)) + min(1,len(ind2)) + min(1,len(ind3)) + min(1,len(ind4))                    
        if summed23 > 3:
            include = 0  
            
           #remove sentences with pneumothorax .
        ind1 = re.findall(r'no ', sentence)
        ind2 = re.findall(r'hypermet', sentence)
        ind3 = re.findall(r'pneumothorax', sentence)
        ind4 = re.findall(r'pleur', sentence)
        summed24 = min(1,len(ind1)) + min(1,len(ind2)) + min(1,len(ind3)) + min(1,len(ind4))                    
        if summed24 > 3:
            include = 0   
            
            #remove sentences with CRANIUM .
        ind1 = re.findall(r'no ', sentence)
        ind2 = re.findall(r'abnorm', sentence)
        ind3 = re.findall(r'cranium', sentence)
        ind4 = re.findall(r'skull', sentence)
        summed25 = min(1,len(ind1)) + min(1,len(ind2)) + min(1,len(ind3)) + min(1,len(ind4))                    
        if summed25 > 3:
            include = 0               
        
             #remove sentences with low-dose ct .
        ind1 = re.findall(r'low-dos ', sentence)
        ind2 = re.findall(r'ct ', sentence)
        ind3 = re.findall(r'not ', sentence)
        ind4 = re.findall(r'suffici', sentence)
        ind5 = re.findall(r'diagno', sentence)
        summed26 = min(1,len(ind1)) + min(1,len(ind2)) + min(1,len(ind3)) + min(1,len(ind4))     + min(1,len(ind5))                 
        if summed26 > 3:
            include = 0               
               
             #remove sentences with teaching .
        ind1 = re.findall(r'teach', sentence)
        ind2 = re.findall(r'physician', sentence)
        ind3 = re.findall(r'examin', sentence)
        summed27 = min(1,len(ind1)) + min(1,len(ind2)) + min(1,len(ind3))                
        if summed27 > 2:
            include = 0    
            
             #remove sentences with require diagn .
        ind1 = re.findall(r'find', sentence)
        ind2 = re.findall(r'requir', sentence)
        ind3 = re.findall(r'diagno', sentence)
        ind4 = re.findall(r'ct ', sentence)
        summed28 = min(1,len(ind1)) + min(1,len(ind2)) + min(1,len(ind3)) + min(1,len(ind4))               
        if summed28 > 3:
            include = 0    
            
             #remove sentences with require diagn .
        ind1 = re.findall(r'not ', sentence)
        ind2 = re.findall(r'adequ', sentence)
        ind3 = re.findall(r'surgic', sentence)
        ind4 = re.findall(r'plan', sentence)
        ind5 = re.findall(r'purpos', sentence)
        summed29 = min(1,len(ind1)) + min(1,len(ind2)) + min(1,len(ind3)) + min(1,len(ind4))   + min(1,len(ind5))              
        if summed29 > 3:
            include = 0    
            
             #remove sentences with require diagn .
        ind1 = re.findall(r'low ', sentence)
        ind2 = re.findall(r'dose', sentence)
        ind3 = re.findall(r'ct', sentence)
        ind4 = re.findall(r'screen', sentence)
        ind5 = re.findall(r'attenu', sentence)
        ind6 = re.findall(r'correl', sentence)
        summed30 = min(1,len(ind1)) + min(1,len(ind2)) + min(1,len(ind3)) + min(1,len(ind4))   + min(1,len(ind5)) + min(1,len(ind6))               
        if summed30 > 4:
            include = 0            
            
             #remove sentences with require diagn .
        ind1 = re.findall(r'requir ', sentence)
        ind2 = re.findall(r'diagno', sentence)
        ind3 = re.findall(r'qualiti', sentence)
        ind4 = re.findall(r'full', sentence)
        ind5 = re.findall(r'breath', sentence)
        ind6 = re.findall(r'thorac', sentence)
        summed31 = min(1,len(ind1)) + min(1,len(ind2)) + min(1,len(ind3)) + min(1,len(ind4))   + min(1,len(ind5)) + min(1,len(ind6))               
        if summed31 > 4:
            include = 0        
       
        if include==1:
            output_sentences.append(sentence)
        
    text = "".join([" "+i if not i.startswith("'") else i for i in output_sentences]).strip()    
       
    return text


def replace_section_headers(text):
    #reports have sections with headers -- this harmonizes/standardizes those
    text = re.sub(r'abdomen pelvi:', 'pelvi_section', text)
    text = re.sub(r'musculoskelet extrem cutaneu:', 'musculoskelet_section', text)
    text = re.sub(r'musculoskelet extrem:', 'musculoskelet_section', text)
    text = re.sub(r'extrem musculoskelet:', 'musculoskelet_section', text)
    text = re.sub(r'cutaneu:', 'musculoskelet_section', text)
    text = re.sub(r'skelet:', 'musculoskelet_section', text)
    text = re.sub(r'bone:', 'musculoskelet_section', text)
    text = re.sub(r'pelvi:', 'pelvi_section', text)
    text = re.sub(r'abdomen:', 'pelvi_section', text)
    text = re.sub(r'head_and_neck:', 'head_and_neck_section', text)
    text = re.sub(r'neck:', 'head_and_neck_section', text)
    text = re.sub(r'chest:', 'chest_section', text)
    text = re.sub(r'musculoskelet:', 'musculoskelet_section', text)
    text = re.sub(r'extrem:', 'extrem_section', text)
    #replace sections starting with . 1. or . 2. , etc
    text = re.sub(r'(\.  )([1-9]\.)', r'\1', text)
    text = re.sub(r'(\. )([1-9]\.)', r'\1', text)
    
    #sometimes they start with 1.   
    if text[0:5].find("1.") > -1:
        text = text.replace('1.','', 1)
    ## get these from the following code                
    # get section headers
#    header = []
#    single_words = all_counts[3].most_common(5000)
#    for word_i in single_words:
#        word = word_i[0]
#        if ':' in word:
#            header.append(word)
   
    text = re.sub(r':', '', text)   #all the rest
    return text










####################################################################################
######                                 Main                             ############
####################################################################################
    
def run(
    #direct = '/home/tjb129/r-fcb-isilon/research/Bradshaw/Lymphoma_UW_Retrospective/Reports',
    direct = 'Z:\Lymphoma_UW_Retrospective\Reports',
    indications_file = 'indications.xlsx',
    mrn_sheet = 'lymphoma_uw_finding_or_impres',
):
    
    synonyms_file = 'synonyms_in_reports.xlsx'
    synonyms_sheet = 'synonyms'
    
    save_file = 'indications_processed.xlsx'
    save_sheet = 'impression_processed'
    
    # read in full data
    df = pd.read_excel(os.path.join(direct, indications_file), mrn_sheet)
    
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
    
    
    
    findings = df['impression']
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
    filtered_findings = remove_low_frequency_words_and_repeated_words(filtered_findings, freq=10)       
    #save
          
    df['impression_processed'] = filtered_findings
    df.to_excel(os.path.join("Z:\\Zach_Analysis\\text_data", save_file), sheet_name=save_sheet)

