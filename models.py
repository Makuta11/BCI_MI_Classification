import torch
from torch import utils
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

# Function for dimension calculation 
def conv2d_output_shape(f_l, kernel_size = 1, stride = 1, padding = 1, dilation=1):
  from math import floor
  if type(kernel_size) is not tuple:
      kernel_size = (kernel_size, kernel_size)
  if type(stride) is not tuple:
      stride = (stride, stride)
  if type(padding) is not tuple:
      padding = (padding, padding)
  f = floor( ((f_l[0] + (2 * padding[0]) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride[0]) + 1)
  l = floor( ((f_l[1] + (2 * padding[1]) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride[1]) + 1)
  
  return f, l

# Model architecture
class CnnNetManyToMany(nn.Module):

    def __init__(self, data_shape, seq_length, conv_hidden_dim, lstm_hidden_dim, fc_hidden_dim,
                 dropout_prop, num_classes):
        
        super(CnnNetManyToMany, self).__init__()
        self.conv_hidden_dim = conv_hidden_dim
        self.layer_img1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = self.conv_hidden_dim[0],
                                kernel_size = (1, 3),
                                stride = (1, 1), 
                                padding = (0, 1)),
           nn.BatchNorm2d(self.conv_hidden_dim[0]),
           nn.ReLU(),  
           nn.MaxPool2d(kernel_size = (1, 2), stride = (1, 2)),
           nn.Dropout(p = dropout_prop)
        )

        #self.layer_img2 = nn.Sequential(
        #    nn.Conv2d(in_channels = self.conv_hidden_dim[0], out_channels = self.conv_hidden_dim[1],
        #                        kernel_size = (1, 3),
        #                         stride = (1, 1),
        #                         padding = (0, 1)),
        #    nn.BatchNorm2d(self.conv_hidden_dim[1]),
        #   nn.ReLU(),
        #   nn.MaxPool2d(kernel_size = (1, 2), stride = (1, 2)),
        #   nn.Dropout(p = dropout_prop)
        #)

        f, l = conv2d_output_shape(data_shape, kernel_size = (1,2), 
                                      stride = (1,2), padding = (0,0))

        #f, l = conv2d_output_shape((f, l), kernel_size = (1,2), 
        #                              stride = (1,2), padding = (0,0))
        
        self.lstm_input = self.conv_hidden_dim[0] * l

        self.batchnorm_2 = nn.BatchNorm1d(seq_length)

        self.lstm = nn.LSTM(input_size = self.lstm_input, hidden_size = lstm_hidden_dim,
                            batch_first = True, bidirectional = False, num_layers = 2)
        self.batchnorm_3 = nn.BatchNorm1d(seq_length) 
        self.dropout = nn.Dropout(p = dropout_prop)
        
        self.fc1 = nn.Linear(in_features = lstm_hidden_dim, out_features = fc_hidden_dim)
        
        self.batchnorm_4 = nn.BatchNorm1d(seq_length)
        
        self.fc2 = nn.Linear(in_features = fc_hidden_dim, out_features = num_classes)

    def forward(self, X_data):
        batch_size = X_data.shape[0]
        time = X_data.shape[1]

        # Convolutional Layers
        X_data = X_data.permute(0, 2, 1, 3)
        X_data = self.layer_img1(X_data)
        #X_data = self.layer_img2(X_data)
        X_data = X_data.permute(0, 2, 1, 3)

        # Flatten for lstm input
        X_data = X_data.reshape(batch_size, time, -1) 
        X_data = self.batchnorm_2(X_data)

        # LSTM layers
        X_data, _ = self.lstm(X_data)
        X_data = self.batchnorm_3(X_data)
        X_data = self.dropout(X_data)
        X_data = F.relu(X_data)

        # Linear Fully Connected Layers
        X_data = self.fc1(X_data)
        X_data = self.batchnorm_4(X_data)
        X_data = self.dropout(X_data)
        X_data = F.relu(X_data)

        X_data = self.fc2(X_data)

        X_data = X_data.permute(0, 2, 1)

        # Output layer for classification
        return F.softmax(X_data, dim = 1) #change dim to 1

# Model architecture
class CnnNetConvLSTM(nn.Module):

    def __init__(self, data_shape, seq_length, conv_hidden_dim, lstm_hidden_dim, fc_hidden_dim,
                 dropout_prop, num_classes):
        
        super(CnnNetConvLSTM, self).__init__()
        self.conv_hidden_dim = conv_hidden_dim
        self.layer_img1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = self.conv_hidden_dim[0],
                                kernel_size = (3, 3),
                                stride = (1, 1), 
                                padding = (1, 1)),
           nn.BatchNorm2d(self.conv_hidden_dim[0]),
           nn.ReLU(),  
           nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2)),
           nn.Dropout(p = dropout_prop)
        )

        self.layer_img2 = nn.Sequential(
            nn.Conv2d(in_channels = self.conv_hidden_dim[0], out_channels = self.conv_hidden_dim[1],
                                kernel_size = (3, 3),
                                 stride = (1, 1),
                                 padding = (1, 1)),
            nn.BatchNorm2d(self.conv_hidden_dim[1]),
           nn.ReLU(),
           nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2)),
           nn.Dropout(p = dropout_prop)
        )

        f, l = conv2d_output_shape(data_shape, kernel_size = (2,2), 
                                      stride = (2,2), padding = (0,0))

        f, l = conv2d_output_shape((f, l), kernel_size = (2,2), 
                                      stride = (2,2), padding = (0,0))
        
        self.flatten_1 = nn.Flatten(start_dim = 2)

        self.lstm_input = self.conv_hidden_dim[1] * l

        self.lstm = nn.LSTM(input_size = self.lstm_input, hidden_size = lstm_hidden_dim,
                            batch_first = True, bidirectional = False, num_layers = 2)
        self.batchnorm_3 = nn.BatchNorm1d(seq_length) 
        self.dropout = nn.Dropout(p = dropout_prop)

        self.flatten_2 = nn.Flatten(start_dim = 1)
        self.fc_layer = nn.Sequential(
            nn.Linear(in_features = lstm_hidden_dim * f , out_features = fc_hidden_dim),
            nn.BatchNorm1d(fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prop),
            nn.Linear(fc_hidden_dim, fc_hidden_dim//5),
            nn.BatchNorm1d(fc_hidden_dim//5),
            nn.ReLU(),
            nn.Dropout(dropout_prop),
            nn.Linear(fc_hidden_dim//5, num_classes)
        ) 

    def forward(self, X_data):
        # Convolutional Layers
        X_data = X_data.permute(0, 2, 1, 3)
        X_data = self.layer_img1(X_data)
        X_data = self.layer_img2(X_data)
        X_data = X_data.permute(0, 2, 1, 3)

        # Flatten for lstm input
        X_data = self.flatten_1(X_data)

        # LSTM layers
        X_data, _ = self.lstm(X_data)
        X_data = self.dropout(X_data)
        X_data = F.relu(X_data)

        # Linear Fully Connected Layers
        X_data = self.flatten_2(X_data)
        X_data = self.fc_layer(X_data)

        # Output layer for classification
        return F.softmax(X_data, dim = 1) #change dim to 1
        