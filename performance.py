import numpy as np
import pandas as pd 

def test_scores(pred, label):
	unique_labels = np.unique(label)
	collect = pd.DataFrame()
	for lab in unique_labels:
		tp = ((pred == lab) & (label == lab)).sum()
		fp = ((pred == lab) & (label!=lab)).sum()
		fn = ((pred != lab) & (label == lab)).sum()
		
		if tp+fp ==0:
			precision = 0
		else:
			precision = tp/(tp+fp)
		
		if tp + fn ==0:
			recall = 0
		else:
			recall = tp/(tp+fn)
		
		if precision + recall == 0:
			f_score = 0
		else:
			f_score = 2*precision*recall/(precision + recall)
		
		temp = pd.DataFrame({'Precision': precision, 'Recall': recall, 'F1 score': f_score}, index = [lab])
		collect = collect.append(temp)
    
	return collect