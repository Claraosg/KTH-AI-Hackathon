
import numpy as np
import pandas as pd

def to_supervised(data):
    x = []
    y = []
    
    ini = 0
    end = 10
    
    while end < data.shape[0]-10+1:
        x.extend(data.iloc[ini:end, 1:].to_numpy())
        y.extend(data.iloc[end:end+10, 1:].to_numpy())
        
        ini += 1
        end += 1
        
    return x, y

def train_test(data):
    x, y = to_supervised(data)
    
    end_point = int(0.8*len(x))
    
    train_x = x[:end_point]
    train_y = y[:end_point]
    
    test_x = x[end_point:]
    test_y = y[end_point:]
    
    return train_x, train_y, test_x, test_y

def undummify(data):

    for index, row in data.iterrows():
        d = {}

        data['Count'][index] = round(data['Count'][index])

        for col in data.columns:

            if col.startswith('Sex'):
                v = round(row[col])
                data[col][index] = 'Male' if v == 0 else 'Female'
            elif col.startswith('Country') and '_' in col:
                splitted = col.split('_')
                country = splitted[1]
                num = int(splitted[0][-1])

                if num not in d.keys():
                    d[num] = {}

                d[num][country] = row[col]
            elif col.startswith('Age'):
                data[col][index] = round(data[col][index])

        for num in d.keys():
            label = 'Country' + str(num)

            best_c = ''
            best_score = -1

            for country in d[num].keys():
                val = d[num][country]

                if val > best_score:
                    best_score = val
                    best_c = country

            if label not in data.columns:
                data[label] = ''

            data[label][index] = best_c
        
    data = data.loc[:,~data.columns.str.contains('_', case=False)] 
    
    return data

def output_format(data, cols):
    data = pd.DataFrame(data)
    data.columns = cols[1:]
    data = undummify(data)
    return data

