# import sys
# sys.modules[__name__].__dict__.clear()

#%% Define the training adn testing data
import numpy as np
import torch
import numpy as np
from LSTM_network import LSTM
import torch.optim as optim
import pandas as pd
from matplotlib import pyplot as plt



output_data = pd.read_csv("Utot_profile.csv",skiprows = None , header = None)
output_data = output_data.values
output_tensor = (torch.from_numpy(output_data)).float()

input_data = pd.read_csv("Time_profile.csv",skiprows = None , header = None)
input_data = input_data.values
input_tensor = (torch.from_numpy(input_data)).float()

list = []

no_list = 20

for i in range(no_list):
    list.append('u_{}'.format(i))
print("list =====", list)



test_step = 5

def val_scale(vec):
    max_val_vec, max_idxs = torch.max(vec, dim=0)
    min_val_vec, min_idxs = torch.min(vec, dim=0)
    scaled_vec =  (vec-min_val_vec)/(max_val_vec-min_val_vec)
    return scaled_vec,max_val_vec,min_val_vec

# Expanded_input_scaled,max_input,min_input = val_scale(Expanded_input_tensor)
# Expanded_output_scaled,max_output,min_output = val_scale(Expanded_output_tensor)

tensors_x = torch.stack([input_tensor[i:(i+test_step)] for i in range(len(input_tensor)-test_step-1)])
tensors_y = output_tensor[test_step+1:]

    
       
input_dimension = 1
output_dimension = 20
hidden_size = 5
num_layers = 1
model = LSTM(input_dimension, output_dimension, hidden_size, num_layers, test_step)
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.1) 



for epoch in range(20000):
# Forward pass
    
    y_pred = model(tensors_x)   
    loss = criterion(y_pred, tensors_y) 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % 1 == 0:
      print('Epoch %d loss: %.7f' % (epoch+1, loss))    
        
prediction = y_pred.detach().numpy()
actual = tensors_y.detach().numpy()       
# plt.plot(np.arange(0,len(prediction)),prediction[:,0],label='Actual')
# plt.plot(np.arange(0,len(actual)),actual[:,0] , label ='Predicted')





# print("indices ===", amplitude_ytensor)      

    

    
    

 
