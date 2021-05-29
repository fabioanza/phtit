import numpy as np
import datagen_Heisenberg as dgen
import indexing as ind
from pathlib import Path
import pickle
import os

class DataGen_Config:
    def __init__(self,L,J_vec,B_vec,bc,del_t,T,psi0,psi0_tag,path):
        #Initialize attributes about the physics of the problem
        self.size=L
        self.J_vec = J_vec
        self.B_vec = B_vec
        self.bc = bc
        self.del_t = del_t
        self.time_steps = T
        self.initial_state = psi0
        self.initial_state_tag = psi0_tag
        assert type(self.initial_state_tag)==str, 'Argument {arg} should be a string'.format(arg=self.initial_state_tag)
        self.initial_state_tag = self.initial_state_tag.lower()
        #Initialize the path where the data goes
        if path[-1]=="/":
            self.folder_path = path
        else:
            self.folder_path = path+"/"
        #Initialize the attributes for the database stuff
        self.path_database_ham = self.folder_path+'ham_database.json'
        assert os.path.exists(self.path_database_ham)==True, "I can not find a database at {here}".format(here=self.path_database_ham)
        self.database_ham = ind.Index(self.path_database_ham)

        self.path_database_eigs = self.folder_path+'eigs_database.json'
        assert os.path.exists(self.path_database_eigs)==True, "I can not find a database at {here2}".format(here2=self.path_database_eigs)
        self.database_eigs = ind.Index(self.path_database_eigs)

        self.path_database_unit = self.folder_path+'unit_database.json'
        assert os.path.exists(self.path_database_unit)==True, "I can not find a database at {here3}".format(here3=self.path_database_unit)
        self.database_unit = ind.Index(self.path_database_unit)

        self.path_database_dyn = self.folder_path+'dyn_database.json'
        assert os.path.exists(self.path_database_dyn)==True, "I can not find a database at {here4}".format(here4=self.path_database_dyn)
        self.database_dyn = ind.Index(self.path_database_dyn)


    def which_database(self,which):
        assert (which=='ham' or which=='eigs' or which=='unit' or which=='dyn')==True ,"Second argument of datagen can only be `ham`, `eigs`, `unit` or `dyn`"
        if which=='ham':
            return self.database_ham
        elif which=='eigs':
            return self.database_eigs
        elif which=='unit':
            return self.database_unit
        elif which=='dyn':
            return self.database_dyn

    def filename(self,pre,identifier):
        return self.folder_path + "{}{}.pickle".format(pre, identifier)

    def database_name(self,which):
        return self.folder_path + "{}_database.json".format(which)

    def print_database(self,which):
        db = self.which_database(which)
        return db.print_database()

    def datagen(self,which):
        #First part: Check database presence.
        if which=='ham' or which=='eigs':
            record_to_check=[self.size,self.J_vec,self.B_vec,self.bc]
            print("Hamiltonian parameters: L = {LL}, J = {JJ}, B = {BB}, bc = {bcbc}".format(LL=self.size,JJ=self.J_vec,BB=self.B_vec,bcbc=self.bc))
        elif which=='unit':
            record_to_check=[self.size,self.J_vec,self.B_vec,self.bc,self.del_t]
        elif which=='dyn':
            record_to_check=[self.size,self.J_vec,self.B_vec,self.bc,self.del_t,self.time_steps,self.initial_state_tag]
        db_name = self.database_name(which)
        print("Looking inside the database {db}".format(db=db_name))
        db = self.which_database(which)
        idx = db.get_label(record_to_check)
        if idx!='N':
            #Record found in the appropriate database. Extract it.
            filename = self.filename(pre=which+"_",identifier=idx)
            print('The data was already generated. Retreiving it at {} ... '.format(filename))
            with open(filename, 'rb') as handle:
                OUTPUT = pickle.load(handle)
            return OUTPUT
        else:
            #Record not in the database. So, we need to build it.
            print("The data was not in the database {dab}. We need to build it.".format(dab=db_name))
            #First, build the class.
            Heisenberg = dgen.Hamiltonian_Heisenberg(self.size,self.J_vec,self.B_vec,self.bc)
            #Then, use the appropriate functions to build the desired output.
            if which=='ham':
                OUTPUT = Heisenberg.build_Hamiltonian()
            elif which=='eigs':
                HAM = self.datagen('ham')
                OUTPUT = Heisenberg.diagonalize_matrix(HAM)
            elif which=='unit':
                HAM = self.datagen('ham')
                Heisenberg_Dynamics = dgen.Dynamics(HAM,self.del_t,self.time_steps,self.initial_state)
                OUTPUT = Heisenberg_Dynamics.propagator(HAM)
            elif which=='dyn':
                UU = self.datagen('unit')
                HAM = self.datagen('ham')
                Heisenberg_Dynamics = dgen.Dynamics(HAM,self.del_t,self.time_steps,self.initial_state)
                OUTPUT = Heisenberg_Dynamics.time_evolution(UU)

            #Now Add record to the database
            print("Adding record to database {dab}".format(dab=db_name))
            db.add_record(record_to_check)
            print("Extracting label of new record and creating the new filename")
            new_idx = db.get_label(record_to_check)
            name_of_file = self.filename(pre=which+"_",identifier=new_idx)
            print("Saving data at {stringa}".format(stringa=name_of_file))
            with open(name_of_file, 'wb') as handle:
                pickle.dump(OUTPUT, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("DONE.")
            return OUTPUT
