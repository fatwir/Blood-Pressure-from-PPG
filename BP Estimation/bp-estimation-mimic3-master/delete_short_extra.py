import scipy.io as sio
import os

# Change the path to that of the master folder
path=r"C:\Users\Fatwir SM\Desktop\Final_Code_GITLAB\BP Estimation\bp-estimation-mimic3-master\models\\"

import os
if os.getcwd()!=path:
  os.chdir(path)

# minimum time
min_time = 600   #sec
frequency = 125  #HZ


patients = [j for j in os.listdir(path +"out\\")]
for patient in patients:
	files = [i for i in os.listdir(path +"out\\"+patient) if '.mat' == i[-4:]]
	print(len(files))
	for filename in files:
	    mat = sio.loadmat(path +"out\\"+patient+"\\" + filename)['val']
	    #Check if its shorter than min time
	    if mat.shape[1] < min_time*frequency:
	    	print('Deleted: ' + filename)
	    	os.remove(path +"out\\"+patient+"\\" + filename)
	    	os.remove(path +"out\\"+patient+"\\" + filename[:-4]+'.hea')
	    	os.remove(path +"out\\"+patient+"\\" + filename[:-5]+'.info')