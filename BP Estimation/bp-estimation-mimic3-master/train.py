# The neccessary imports
import os,re,sys,keras
import numpy as np
import pandas as pd
from sklearn.model_selection  import LeaveOneOut

# Change the path to the folder where you wish to save the weights files and the model
dir = r"C:\Users\Fatwir SM\Desktop\Final_Code_GITLAB\BP Estimation\bp-estimation-mimic3-master\models\\"
 
# Change the path to the folder where the records have been downloaded
path = r"C:\Users\Fatwir SM\Desktop\Final_Code_GITLAB\BP Estimation\bp-estimation-mimic3-master\models\final\\"

# Change the path to where the model.py code is saved
path_1 = r"C:\Users\Fatwir SM\Desktop\Final_Code_GITLAB\BP Estimation\bp-estimation-mimic3-master\models\\"

# Extracting the data
# Change the path to the folder where the records have been downloaded

if os.getcwd()!=path:
  os.chdir(path)
  sys.path.insert(0, path)
     
dataset = list()

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
          
          # Getting the number of columns
          column = gg.shape[1]

          # Appending this read record to the main dataset
          dataset.append(pd.read_csv(sub_record))

    # Getting back to the initial folder      
    if os.getcwd()!=path:
      os.chdir(path)
      sys.path.insert(0, path)

# I have trained the model for 5 patient records
# Comment the below line to train for all the patient records
dataset = dataset[0:5]

# Remove the NaN values from the dataset

for i in range(0,len(dataset)):
  dataset[i] = dataset[i].interpolate()

# Training the model

# Changing the path to where the models.py file is located

if os.getcwd()!=path_1:
  os.chdir(path_1)
  sys.path.insert(0, path_1)


# Imports
from models import *
from models import raw_signals_deep_ResNet,custom_callback

# Index to define the model only in the first iteration
model_def_index = 1

# Initializing the final result vector
result_final = np.zeros([len(dataset),2])

# Index initialization
result_index = 0

# Calling the leave one out CV function
loo = LeaveOneOut()

# Training the model using LOSO CV

for train_index, test_index in loo.split(dataset):

    # Getting the input dataset and defining the no. of rows and columns
    input_train_raw = np.array([dataset[i] for i in train_index])
    
    no_row = np.sum(input_train_raw[i].shape[0] for i in range (0,input_train_raw.shape[0]))  
    no_column = input_train_raw[0].shape[1]  

    input_formatted = np.zeros([no_row,no_column])
    start = 0
    
    for k in range (0,input_train_raw.shape[0]):
          
          input_initial = input_train_raw[k]
          input_initial = input_initial.to_numpy()
          size = input_initial.shape[0]
          input_formatted[start:start+size,:] = input_initial
          start = size


    # Training Input and Output
    # The second last 2 columns of the records are the SBP and DBP values (the outputs)
    # From the first column till the SBP column is the input recording
        
    input_train = input_formatted[:,0:no_column-4]
    output_train = input_formatted[:,no_column-4:no_column-2]

    input_test_raw = np.array([dataset[i] for i in test_index])
    
    input_test = input_test_raw[0,:,0:no_column-4]
    output_test = input_test_raw[0,:,no_column-4:no_column-2]

    # # The size of the input training and testing data keeps changing as is expected
    # # Uncomment the below lines to see this
    # print(input_formatted.shape)
    # print(input_test_raw[0].shape)

    # Computing the derivatives of the training inputs
    
    input_train_diff_1 = np.diff(input_train)
    input_train_diff_2 = np.diff(input_train,n=2)
    
    # Appending zeros to the derivatives to get the same size as that of the training inputs

    input_train_diff_1 = np.insert(input_train_diff_1,0,0,axis=1)
    input_train_diff_2 = np.insert(input_train_diff_2,0,0,axis=1)
    input_train_diff_2 = np.insert(input_train_diff_2,0,0,axis=1)
     
    # Computing the derivatives of the testing inputs

    input_test_diff_1 = np.diff(input_test)
    input_test_diff_2 = np.diff(input_test,n=2)
    
    # Appending zeros to the derivatives to get the same size as that of the testing inputs
    
    input_test_diff_1 = np.insert(input_test_diff_1,0,0,axis=1)
    input_test_diff_2 = np.insert(input_test_diff_2,0,0,axis=1)
    input_test_diff_2 = np.insert(input_test_diff_2,0,0,axis=1)
    
    # Performing some necessary reshaping operations

    input_train = np.reshape(input_train,(input_train.shape[0],input_train.shape[1],1))
    input_train_diff_1 = np.reshape(input_train_diff_1,(input_train_diff_1.shape[0],input_train_diff_1.shape[1],1))
    input_train_diff_2 = np.reshape(input_train_diff_2,(input_train_diff_2.shape[0],input_train_diff_2.shape[1],1))
    
    input_test = np.reshape(input_test,(input_test.shape[0],input_test.shape[1],1))
    input_test_diff_1 = np.reshape(input_test_diff_1,(input_test_diff_1.shape[0],input_test_diff_1.shape[1],1))
    input_test_diff_2 = np.reshape(input_test_diff_2,(input_test_diff_2.shape[0],input_test_diff_2.shape[1],1))
    
    output_train = np.reshape(output_train,(output_train.shape[0],output_train.shape[1]))
    output_test = np.reshape(output_test,(output_test.shape[0],output_test.shape[1]))
    
    # Defining the model for the first iteration
     
    if (model_def_index==1):
        
        # Updating the model index so that re initialization of the model does not occur
        model_def_index = 0
        # Defining the model
        model = raw_signals_deep_ResNet([int(input_train.shape[1]),1],3)
    
    # Testing and training indices    
    print("TRAIN:", train_index, "TEST:", test_index)
      
    #Train the model
    
    model_name = "Final Model"
   
    # Early stopping
    es = keras.callbacks.EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 3)  
    
    batch_size = 128
    
    history=model.fit([input_train,input_train_diff_1,input_train_diff_2],output_train, epochs = 20,
                      shuffle = True, batch_size = batch_size , callbacks=[custom_callback(dir,model_name),es],validation_split = 0.2)
     
    # Test Accuracy

    result = model.predict([input_test,input_test_diff_1,input_test_diff_2])
    result_final[result_index,:] = np.mean(np.abs(result-output_test),axis=0)
    
    print('\nSBP Test MAE: ', result_final[result_index,0])
    print('\nDBP Test MAE: ', result_final[result_index,1])
    
    # Updating the result index

    result_index+=1

# Save the model after training
model.save(dir + "model")


# Final Average of the MAE values after training and testing
final_result = np.mean(np.abs(result_final),axis=0)

print('Final SBP Test MAE: ', final_result[0])
print('\nFinal DBP Test MAE: ', final_result[1])    
     
