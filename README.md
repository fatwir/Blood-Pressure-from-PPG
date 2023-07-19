### Data Retrieval
*NOTE: The shell commands are included in the file named "prereq.txt". These need to be copied and executed from the terminal. Please ensure that all the packages present in "requirements.txt" are included in the system. Also, Python version 3.7 is needed to run the python scripts*
1.	You will need to download the WFDB software package from physiotools/wfdb.tar.gz. I have downloaded the latest version and have included the same in the directory (this worked in my Windows 10 64 bit operating system)
2.	This file needs to be unzipped and installed on the local machine.
3.	Open DataMiner.sh file and change the path of the folder to which the records will be downloaded in.
4.	Run this bash script file from the terminal.
5.	Records will start downloading, and this could take some time
6.	Once these files have been downloaded, run the DataCleaner.sh file
7.	After this, run the organize_patients.py file. This will organize all the patient's records into a folder named "Out".
8.	Open "delete_short_extra.py" and change the path (details are present in the .py file) accordingly. Run this script, and certain records will be deleted.
9.	Further processing of the records will be done using MATLAB scripts.
### Data Pre-Processing
1.	In the folder named "cleaning_scripts" present in the main folder, open the "main.m" file using MATLAB. Once opened, change the Input Directory (dirName) and Output Directory (testOutDir) for your machine.
2.	Remember to add the path of the cleaning_scripts folder to MATLAB since there are functions present in this folder that will be needed to run "main.m"
3.	The Input Directory is the place where the records had been downloaded and are now stored. The Output Directory is where the records after processing will be stored.
4.	Run this .m script file, and data cleaning should start. MATLAB will print outputs with each stage of record cleaning. Once after all records have been cleaned, there will also be some plots that will be generated.
5.	This completes data retrieval and pre-processing

*A subset of data comprising the records of 17 patients has been downloaded and pre-processed*
### Training the Spectro Temporal Deep Network
1.	Open train.py and change the respective directories (details are present in the .py file)
2.	Comment line 58 if you wish to train for the entire set of records (I have trained for five patient records)
3.	Open models.py and change the directory (details are present in the .py file)
4.	Training was done on Google Colab, hence only 1 GPU was used. If you wish to use more than 1 GPU, please uncomment line 130 of models.py
5.	Run train.py. Training should start, and once the model starts performing better than a threshold, the weights file will be saved
### Testing the Spectro Temporal Deep Network
1.	Open test.py and change the appropriate paths (details are present in the .py file)
2.	Change the number of test records if needed (I have randomly selected 500 samples to test the network on)
3.	Run this file and testing should take place 


