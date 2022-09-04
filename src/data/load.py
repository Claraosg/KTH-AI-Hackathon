

import numpy as np
import pandas as pd
from datetime import datetime, date
from dateutil.relativedelta import relativedelta

def compute_age(birth, year_award):
    birth_date = datetime.strptime(str(birth), '%Y-%m-%d')
    award_date = datetime(int(year_award),12,10,0,0,0,0) # day when Nobel prize is awarded every year
    
    difference = relativedelta(award_date, birth_date)
    return difference.years

def combine_winners(df):
    for year in df['Year'].unique():
        tmp = (df[df['Year'] == year])
        
        c = tmp.iloc[0,4]
            
        df = df[df['Year'] != year]

        ages = tmp['Age'].to_numpy()
        countries = tmp['Birth Country'].to_numpy()
        genders = tmp['Sex'].to_numpy()
        
        entry = {'Year':year, 'Count':c}
    
        for j in range(5):
            i = j % len(ages)
            entry['Age'+str(j)] = ages[i]
            entry['Country'+str(j)] = countries[i]
            entry['Sex'+str(j)] = 0 if genders[i] == 'Male' else 1
            
        df = df.append(entry, ignore_index=True)

    return df    

def load_data(category='Medicine'):
    
    # load csv
    data = '../dataset/archive.csv'
    df = pd.read_csv(data)
    
    # select relevant features
    features = ['Year', 'Category', 'Birth Date', 'Birth Country', 'Sex']
    df = df[features]
    
    # select category
    df = df[df['Category'] == category]
    #df.reset_index()
    
    # compute age from date info
    df['Age'] = df.apply(lambda row: compute_age(row[2], row[0]), axis=1)
    
    # remove 'Birth Date' and 'Category' features    
    df = df[['Year', 'Birth Country', 'Sex', 'Age']]
    
    # create count column
    df['Count'] = df.groupby(['Year'])['Age'].transform("count")
    
    # combine winners of the same year
    df = combine_winners(df)    
    
    # drop old columns    
    df.drop(labels=['Birth Country', 'Sex', 'Age'], inplace=True, axis=1)
    
    # dummify the categorical columns
    df = pd.get_dummies(df)

    return df    

