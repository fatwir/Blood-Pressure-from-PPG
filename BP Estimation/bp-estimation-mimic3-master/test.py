# The neccessary imports
import os,re,sys,keras
import numpy as np
import pandas as pd
import random

# Change the path to where the model.py code is saved
path_1 = r"C:\Users\Fatwir SM\Desktop\Final_Code_GITLAB\BP Estimation\bp-estimation-mimic3-master\models\\"

# Change the path to the folder where the records have been downloaded
path = r"C:\Users\Fatwir SM\Desktop\Final_Code_GITLAB\BP Estimation\bp-estimation-mimic3-master\models\final\\"

# Change this to the number of records you wish to test on
no_records_test = 500

# Getting the records to be tested

if os.getcwd()!=path:
  os.chdir(path)
     
dataset = list()
sum = 0

# Loop to iterate over the records and extract only those that are relevant

# Defining the directory
dirs = sorted(os.listdir())

for record_counter in range(0,len(dirs)):
    # Change the path and check all the files there
    path_new = path + str(dirs[record_counter]) + "\\"
    
    if os.getcwd()!=path_new:
       os.chdir(path_new)
       sys.path.insert(0, path_new)
    
    dirs_1 = sorted(os.listdir())

    for sub_record in dirs_1:
        if re.search("m.mat_matrixRawCyclesGood",sub_record):
          
          # Reading the .csv file
          gg = pd.read_csv(sub_record)
          
          # Getting the total number of rows
          sum+=gg.shape[0]

          # Getting the number of columns
          column = gg.shape[1]

          # Appending this read record to the main dataset
          dataset.append(pd.read_csv(sub_record))

    # Getting back to the initial folder      
    if os.getcwd()!=path:
      os.chdir(path)
      sys.path.insert(0, path)

# Remove the NaN values

for i in range(0,len(dataset)):
  dataset[i] = dataset[i].interpolate()

# Getting the testing data

start = 0
input = np.zeros([sum,column])
 
dataset = np.array(dataset)

for i in range (0,dataset.shape[0]):

   input_initial = dataset[i]
   input_initial = input_initial.to_numpy()
   size = input_initial.shape[0]
   input[start:start+size,:] = input_initial
   start = size

# Now we will randomnly select 500 records to test on
sample_values = random.sample(range(0, input.shape[0]), no_records_test)

# Defining the input and output values for testing
input_test = input[sample_values,0:column-4]
output_test = input[sample_values,column-4:column-2]

# Computing the derivatives
input_test_diff_1 = np.diff(input_test)
input_test_diff_2 = np.diff(input_test,n=2)

input_test_diff_1 = np.insert(input_test_diff_1,0,0,axis=1)
input_test_diff_2 = np.insert(input_test_diff_2,0,0,axis=1)
input_test_diff_2 = np.insert(input_test_diff_2,0,0,axis=1)

# Performing the necessary re-shaping operations
input_test = np.reshape(input_test,(input_test.shape[0],input_test.shape[1],1))
input_test_diff_1 = np.reshape(input_test_diff_1,(input_test_diff_1.shape[0],input_test_diff_1.shape[1],1))
input_test_diff_2 = np.reshape(input_test_diff_2,(input_test_diff_2.shape[0],input_test_diff_2.shape[1],1))
    
output_test = np.reshape(output_test,(output_test.shape[0],output_test.shape[1]))
  
# Changing the path to where the models.py file is located

if os.getcwd()!=path_1:
  os.chdir(path_1)
  sys.path.insert(0, path_1)

# Imports
from models import *
from models import raw_signals_deep_ResNet,custom_callback

# Defining the model 

model = raw_signals_deep_ResNet([int(input_test.shape[1]),1],3)

# Loading the model and testing
model.load_weights(path_1 + "Final Model_weights.h5")
      
# Predicting the values
result = model.predict([input_test,input_test_diff_1,input_test_diff_2])

final_result = np.mean(np.abs(result-output_test),axis=0)

print('\nSBP Test MAE: ', final_result[0])
print('\nDBP Test MAE: ', final_result[1])   
