# Import required libraries
import pandas as pd
import numpy as np 
import sklearn
from sklearn.model_selection import train_test_split
# Keras specific
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical 

import fcsparser
import os 

'''Import data files'''


names = ["AA", "OD", "PC", "SR", "VG"]
dilutions = {"A":0.0016, "B":0.008,"C":0.07,"D":0.2,"E":1}
media = {"01":"P","02":"S","03":"A","04":"V","05":"O","06":"T",
"07":"P","08":"S","09":"A","10":"V","11":"O","12":"T"}
copper = {"01":"copper","02":"copper","03":"copper","04":"copper","05":"copper",
"06":"copper","07":"pure","08":"pure","09":"pure","10":"pure","11":"pure","12":"pure"}

for name in names:

    for file in os.listdir("./crossfeeding/5fcs/" + name + "/"):

        if file.endswith(".fcs"):
            dummystring = str('./crossfeeding/5fcs/'+ name + "/"+ file)
            meta, data = fcsparser.parse(dummystring, meta_data_only=False, reformat_meta=False)
            file = file[:-4]
            pd_data = pd.DataFrame(data)
            dummystring2 = str("./crossfeeding/5fcs/"+ name + "/"+ file + ".csv")
            pd_data.to_csv(dummystring2)
            print(file)


dfs = []
for name in names:

    for file in os.listdir("./crossfeeding/5fcs/" + name + "/"):
        if file.endswith(".csv"):
            dummystring = str('./crossfeeding/5fcs/'+ name + "/" + file)
        
            file = file[:-4]
            row = file[0]
            col = file[1:]
            print(dummystring)
            d = pd.read_csv(dummystring)
            d['well'] = [file for i in range(d.shape[0])]
            d['species'] = [name for i in range(d.shape[0])]
            d['dilution'] = [ dilutions[row] for i in range(d.shape[0])]
            d['media'] = [ media[col] for i in range(d.shape[0])]
            d['copper'] = [ copper[col] for i in range(d.shape[0])]
            dfs.append(d)

fivesdf = pd.concat(dfs)
fivesdf.to_csv("./crossfeeding/5fcs/fiveS.csv")
