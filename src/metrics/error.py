
import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score

import sys
sys.path.append("../..")
from src.data.format import output_format, undummify

def apply_metrics(y_hat, y_actual, cols):
	y_hat = output_format(y_hat, cols)
	y_actual = output_format(y_actual, cols)

	return compute_error(y_hat, y_actual)	

def compute_error(y_hat, y_actual):
    errors = pd.DataFrame(columns=y_hat.columns)
    
    entry = {}
    
    # MSE for numerical data
    # Accuracy for categorical data
    for col in y_hat.columns:
        if col.startswith('Age') or col == 'Count':
            entry[col] = mean_squared_error(y_actual[col], y_hat[col])
        else: 
            entry[col] = accuracy_score(y_actual[col], y_hat[col]) 
    
    errors = errors.append(entry, ignore_index=True)
    
    return errors
