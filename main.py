import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from dataloader import *
from train_function import *
from models import *
from performance import *

# Perform test train split on 5 subjects
#Users = np.arange(6)
#users_train, users_test = train_test_split(Users, shuffle = True)

# Only used for testing. Remove for final hand in
users_train = [0, 1, 3, 4]
users_test = [2]

# Load data - specify if you would like to load evaluation data as well
data_train, data_test, label_train, label_test = load_data(users_train, users_test)
print("loaded data")

# Data parameters
SEQ_CHANNELS = 3
SEQ_FILTERS = 6
SEQ_LENGTH = 60
NUM_CLASSES = 3
DATA_SHAPE = SEQ_LENGTH, SEQ_FILTERS

# Training parameters
BATCH_SIZE = 12000
NUM_EPOCHS = 1
DROPOUT_PROP = 0.25
LEARNING_RATE = 1e-5
NUM_CLASSES = 3
FC_HIDDEN_DIM = 512
CONV_FILTERS = [32]
LSTM_HIDDEN_DIM = 64

# Create dataset and dataloaders
train_dataset = ImageTensorDatasetMultiEpoch(data_train, users_train, filter_seq = SEQ_LENGTH - 1, label = label_train)
test_dataset = ImageTensorDatasetMultiEpoch(data_test, users_test, filter_seq = SEQ_LENGTH - 1, label = label_test)
print("created datasets")

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
print("created dataloaders")

del train_dataset, test_dataset, data_train, data_test, label_train, label_test
torch.cuda.empty_cache()

# Create device for GPU compatability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
model = CnnNetManyToMany(DATA_SHAPE, SEQ_LENGTH, CONV_FILTERS, LSTM_HIDDEN_DIM, FC_HIDDEN_DIM, DROPOUT_PROP, NUM_CLASSES).to(device)
optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay = 1e-2)
criterion = nn.CrossEntropyLoss()
print("initialized model")

# Run training
loss_collect, val_loss_collect, model = train_model(model, optimizer, criterion, NUM_EPOCHS, train_dataloader, test_dataloader, device, scheduler = None)	
print("trained model")

# Save model for later evaluation
torch.save(model.state_dict(), 'outputs/SeqSeqModel_seq60_batch1200_epoch5_sd.pt')
print("saved model")

# Define prediction evaluation parameters
collect = dict()
collected_data = dict()
predictions = pd.DataFrame()

# Place prediction for each datapoint in a dictionary
for i, x in enumerate(test_dataloader):
	model.eval()
	outList = model(x[0].float().to(device)).detach().cpu()
	outID = x[2]

	outLabelsTrue = x[1].long()
	
	for k in range(outList.shape[0]):
		line = outList[k,:,:]
		ids = outID[k,:].numpy()
		j = 0
		for id in ids:
			if id in collect.keys():
				temp = collect[id]['pred']
				data = line[:,j].unsqueeze(1)
				temp = torch.cat((temp, data), dim = 1)
				collect[id]['pred'] = temp
			else:
				collect[id] = dict()
				collect[id]['label'] = outLabelsTrue[k,j]
				collect[id]['pred'] = line[:,j].unsqueeze(1)
		j+=1

# Calculate log probability for each datapoint and choose max for classification prediction
for key in collect.keys():
	out = collect[key]['pred']
	out = torch.log(out).sum(dim = 1)
	prob, pred = torch.max(out, dim = 0)

	temp = pd.DataFrame({'pred': pred, 'label': collect[key]['label'].numpy()}, index = [key])
	predictions = predictions.append(temp)

# Calculate the scores for model performance
scores = test_scores(predictions['pred'], predictions['label'])
collected_data['loss_collect'] = loss_collect
collected_data['val_loss'] = val_loss_collect
collected_data['scores'] = scores

# Save model performance statistics to a pickle file
with open('modelSaves/ModelOutput_seq60_batch1200_epoch5.pickle', 'wb') as handle:
	pickle.dump(collected_data, handle)
