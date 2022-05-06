# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 09:30:03 2022

@author: beara
"""
import os
import numpy as np
import pandas as pd
import re


def containString(context, key):
    context = context.lower()
    key = key.lower()
    if context.find(key) > -1:
        return True
    else:
        return all(k in context for k in key.split())

def lineSplit(line, sep=':'):
    res = []
    s =''
    for item in line:
        if item != sep:
            s+=item
        else:
            res.append(s.strip())
            s = ''
    if len(s)>0:
        res.append(s.strip())
    output = []
    for item in res:    
        if " "*3 in item:
            items = item.split(" "*3)
            for item in items:
                if len(item.strip())>0:
                    output.append(item.strip())
        else:
            output.append(item.strip())
    return output


def parse_data(context):
    
    context = context.replace('discharge diagnoses:', 'discharge diagnosis:')
    context = context.replace('discharge studies:', 'discharge study:')
    context = context.replace('final diagnoses:', 'final diagnosis:')
    
    lines = context.split('\n')
    lines = [line for line in lines if len(line)>0]
    

    keyWords = ['Admission Date', 'Discharge Date',
                'Date of Birth', 'Sex', 
                'Service', 'Chief Complaint', 'General',
                'HISTORY OF PRESENT ILLNESS',
                'PAST MEDICAL HISTORY',
                'MEDICATIONS ON ADMISSION',
                'ALLERGIES',
                'FAMILY HISTORY',
                'SOCIAL HISTORY',
                'PHYSICAL EXAM',
                'LABORATORY STUDY',
                'BRIEF HOSPITAL COURSE',
                'Discharge Diagnosis',
                'Discharge Disposition',
                'DISCHARGE CONDITION',
                'DISCHARGE STATUS',
                'DISCHARGE MEDICATIONS',
                'Discharge Instructions',
                'FOLLOW-UP PLANS',
                'FINAL DIAGNOSIS']
    keyWords = [kwd.lower() for kwd in keyWords]
    
    keyMapping = dict(zip(keyWords, [None]*len(keyWords)))
    # keys in the data might not be normalized
    # find actual key words in the data
    for line in lines:
        nKeys = line.count(':')
        if nKeys == 0:
            continue
        for keyWord in keyWords:
            if keyMapping[keyWord]:
                continue
            # check if keywords in the line
            if containString(line, keyWord):
                lineItems = lineSplit(line)
                for i, item in enumerate(lineItems):
                    if containString(item, keyWord): 
                        #keyMapping[keyWord] = item[item.lower().find(keyWord.lower()):]
                        keyMapping[keyWord] = lineItems[i].strip()
                        break
                
    for key in keyMapping:
        if keyMapping[key] is None:
            keyMapping[key] = key        
    
    # text: text corresponding to sections in the text
    text = dict(zip(keyWords, [None]*len(keyWords)))
    
    for _key in keyWords:
        try:
            keyWord = keyMapping[_key].lower().split()
            startLine, endLine = None, None
            for line in lines:
                if all([w in line.lower() for w in keyWord]):
                     startLine = line
                     continue
                if startLine is not None and ':' in line:
                    endLine = line
                if endLine is not None:
                    break
            if startLine and endLine:
                dataLine = ' '.join(lines[lines.index(startLine):lines.index(endLine)])
                if dataLine.count(":") > 1:
                    loc = dataLine.find(keyMapping[_key])
                    if loc > -1:
                        dataLine = dataLine[loc:].split(':')[1].strip().split()
                        if len(dataLine) == 0:
                            dataLine = ''
                        else:
                            dataLine = dataLine[0]
                dataLine = dataLine.replace(_key, '').replace(':',' ').strip().replace('[**','').replace('**]','')
                if _key == 'date of birth':
                    if not dataLine.replace('-','').replace('/','').isnumeric():
                        dataLine = ''
                
                text[_key] = dataLine
            else:
                text[_key] = ''
        except:
             text[_key] = ''   
                
    # birth date if faked in the data, need to search for age
    try:
        text['age'] = round(pd.to_datetime(text['admission date']).year - pd.to_datetime(text['date of birth']).year)
    except:
        try:
            text['age'] = round(pd.to_datetime(text['discharge date']).year - pd.to_datetime(text['date of birth']).year)
        except:
            text['age'] = None
            
    normalized_context = context.replace('-', ' ')
    normalized_context = re.sub('\s+',' ',normalized_context)
        
    if text['age'] is None:
        
        age_key_word = ['year old', 'years old', 'yo']
        for item in age_key_word:
            try:
                age = normalized_context[normalized_context.index(item)-3: normalized_context.index(item)].strip()
                if age.isnumeric():
                    text['age'] =  int(age)
                else:
                    if age[1:].isnumeric():
                        text['age'] =  int(age[1:])
                break
            except:
                continue
    if text['age'] is None:
        try:
            loc = normalized_context.find('age over')
            age = normalized_context[loc+len('age over')+1: loc+len('age over')+3]
            text['age'] = int(age.strip())
        except:
            pass
    
    if text['age'] is None:
        # two digits follow f or m
        try:
            age, sex = re.findall(r'\s(\d{2})(m|f)\s', normalized_context)[0]
            text['age'] = int(age.strip())
            text['sex'] = sex
        except:
            pass
    

    if text['sex'] is None or len(text['sex'])<1:
        # need to find gender from text
        gender_context = normalized_context.replace('.','').split()
        menIndicator = any(w in gender_context for w in ['male', 'man', 'men', 'mr', 'he', 'his']) 
        womenIndicator = any(w in gender_context for w in ['female', 'woman', 'women', 'mrs', 'she', 'her']) 
        
        if menIndicator and not womenIndicator:
            text['sex'] = 'm'
        if not menIndicator and womenIndicator:
            text['sex'] = 'f'
     
    if text['sex'] is None or len(text['sex'])<1:
        # regular expression
        # 87 yo m
        try:
            age, _, sex = re.findall(r'\s(\d{2})\s(yo)\s(m|f)', normalized_context.replace('.',''))[0]
            if sex in ('f','m'):
                text['sex'] = sex
            if text['age'] is None:
                text['age'] = int(age)
        except:
            pass
 
    return text


def getRawData(workingDir=r"C:\Users\beara\Desktop\CS598\Project",
               mimicDir=r'mimic-iii-clinical-database-1.4',
               outputDir='data'):
    
    data = pd.read_csv(os.path.join(workingDir, mimicDir, 'NOTEEVENTS.csv'))
    
    
    col = 'TEXT'
    parsed_data_list = []
    for i, text in data.iterrows():
        if i % 1000 == 0 and i > 0:
            print("proccessed", i, "out of", data.shape[0], "EHR records.")
        context = text[col].lower()
        parsed_data = pd.DataFrame(parse_data(context), index=[0])
        parsed_data_list.append(parsed_data)
        
    parsedText = pd.concat(parsed_data_list).reset_index(drop=True)
    parsedText = pd.concat([data, parsedText], axis=1)
    parsedText.drop(columns=['TEXT'], inplace=True)
    
    parsedText.to_csv(os.path.join(workingDir, outputDir, 'parsed_data.csv'), index=False)
    
    
    
    # merge ADMISSIONS
    admission = pd.read_csv(os.path.join(workingDir, mimicDir, 'ADMISSIONS.csv'), 
                            usecols=['SUBJECT_ID','HADM_ID', 'ADMITTIME', 'DISCHTIME',
                                     'ADMISSION_TYPE', 'ADMISSION_LOCATION',
                                     'DISCHARGE_LOCATION', 'MARITAL_STATUS', 
                                     'ETHNICITY', 'DIAGNOSIS'])
    # merge ICD9
    icd9 = pd.read_csv(os.path.join(workingDir, mimicDir, 'DIAGNOSES_ICD.csv'))
    icd9.drop(columns='ROW_ID', inplace=True)
    # process icd9
    icd9 = icd9.groupby(['SUBJECT_ID','HADM_ID']).apply(lambda x: pd.Series(
        {'ICD9_CODE': ";".join(x.ICD9_CODE.astype(str).tolist())}
        )).reset_index()
    
    # merge patient
    patient = pd.read_csv(os.path.join(workingDir, mimicDir, 'PATIENTS.csv'),
                          usecols= ['SUBJECT_ID', 'GENDER', 'DOB'])
    

    
    dataAll = parsedText.merge(admission, on =['SUBJECT_ID', 'HADM_ID'], how='left')
    dataAll = dataAll.merge(icd9, on =['SUBJECT_ID', 'HADM_ID'], how='left')
    dataAll = dataAll.merge(patient, on =['SUBJECT_ID'], how='left')
    dataAll = dataAll.replace({'': None})
    dataAll = dataAll.replace({np.nan: None})

    # DOB
    dataAll.DOB = pd.to_datetime(dataAll.DOB).dt.strftime('%Y-%m-%d')
    dataAll.DOB = [v1 if v1 is not None else v2 for v1,v2 in zip(dataAll.DOB, dataAll['date of birth'])]
    # GENDER
    dataAll.GENDER = dataAll.GENDER.str.lower()
    dataAll.GENDER = [v1 if v1 is not None else v2 for v1,v2 in zip(dataAll.GENDER, dataAll.sex)]
    
    dataAll.drop(columns=['ROW_ID','CGID','ISERROR', 'CHARTTIME', 'STORETIME', 'sex', 'date of birth'], 
                 inplace=True)
    
    # impute age
    age = round(pd.to_datetime(dataAll['ADMITTIME']).dt.year - pd.to_datetime(dataAll['DOB']).dt.year)
    dataAll['age'] = [v1 if v1 is not None else v2 for v1, v2 in zip(dataAll.age, age)]
    dataAll['age'] = dataAll['age'].astype(str)
    
    # save data to disk
    dataAll.to_csv(os.path.join(workingDir, outputDir, 'parsed_data_all.csv'), index=False)


