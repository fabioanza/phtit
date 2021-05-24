import numpy as np
import pandas as pd

class Index:
    def __init__(self,path_name):
        #This is the path with the name of the file containing the index of the data that already exists.
        self.filename = path_name
        self.index_dframe = pd.read_json(self.filename,orient='split')
        self.count = self.index_dframe['Label'].max()
        self.paras_num = len(self.index_dframe.columns)-1

    def check_presence(self,to_check):
        #First, check the formatting of to_check.
        #It would be better to also check the internal formatting. TO DO.
        dropped_label = self.index_dframe.drop('Label',1)
        assert len(to_check)==len(dropped_label.columns), "The record in input has the wrong number of columns. Got {num1}, Expected {num2}".format(num1=len(to_check),num2=len(dropped_label.columns))

        #Initialize a flag to 'N' for No. It will turn to 'Y' if we find the record.
        flag='N'
        for k in dropped_label.index.values.tolist():
            AA = []
            for j in range(len(dropped_label.columns)):
                if type(dropped_label.loc[k][j])==type('A'):
                    #Here it's a string element. So, we can use == to compare.
                    #We need to apply lower(), to avoid confusion with capital letters.
                    AA.append(dropped_label.loc[k][j].lower()==to_check[j].lower())
                else:
                    AA.append(np.allclose(dropped_label.loc[k][j],to_check[j]))
            if all(AA)==True:
                flag='Y'
                val = k
                break
        if flag=='Y':
            return True, val
        else:
            return False

    def get_label(self,to_check):
        #Returns 'N' if the record is not there.
        #Returns the label of the record if the record is actually there
        if self.check_presence(to_check)==False:
            return 'N'
        else:
            _, val = self.check_presence(to_check)
            return self.index_dframe.loc[val]['Label']

    def print_database(self):
        return self.index_dframe

    def add_record(self,new_record): #Adds a record, if we added some data to the folder.
        #Check if the record is already present
        if self.check_presence(new_record)==False:
            print("This record was not present. Adding new record.")
            #the counter is necessary to give a name to the files containing the data
            new_label = self.count+1 #The new label is the larger existing label + 1
            full_new_record = new_record[:] #Remember, if you don't use the slicing operator it remains the same list. We need a different obj, with the same elements.
            full_new_record.append(new_label)
            to_add = pd.DataFrame([full_new_record],columns=self.index_dframe.columns) #
            self.index_dframe = self.index_dframe.append(to_add,ignore_index=True) #Add a row (to the dataframe) with the parameters of the new record.
            print("Updating database with new information")
            self.index_dframe.to_json(self.filename,orient='split',index=False) #Update the json database
            self.count = new_label #update the counter
        else:
            print("This record is already present. Not adding anything.")


    def remove_record(self,old_record): #Allows to remove a record, if we erased the file from the folder.
        #The counter is not touched here. We simply want it to increase when we add new files.
        #Check if the record is present in the databse
        if self.check_presence(old_record)==False:
            print("This record is not present. Nothing to remove.")
        else:
            print("This record is present. Now removing the record.")
            _, idx_to_erase = self.check_presence(old_record)
            self.index_dframe = self.index_dframe.drop(idx_to_erase) #Erase the record from dataframe
            print("Updating index with new information")
            self.index_dframe.to_json(self.filename,orient='split',index=False) #Update the json database
