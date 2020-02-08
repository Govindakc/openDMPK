# ********************************* #
# Govinda KC                        #
# UTEP, Computational Science       #
# Last modified: 2/8/2020           #
# ********************************* #

# Usage: python run_opendmpk.py SMILES ( eg. python run_opendmpk.py "CC(O)CO" )

import os, sys, joblib, time, json
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
from descriptors import Features_Generations

class openDMPK:
    MODELS_DIR = os.path.join('models')

    def __init__(self):
        self.load_models()

    def load_models(self):
        
        with open('models.txt', 'r') as f:
            models = f.read().splitlines()

        self.model_names = [model_path.split('_')[0] for model_path in models]
        self.models = [joblib.load(os.path.join(self.MODELS_DIR, model_path)) for model_path in models]

    def get_features(self, smiles):
        fg = Features_Generations(smiles)
        features = fg.molecule_descriptors()
        return features

    def predict(self, smiles):
        features = self.get_features(smiles)
        final_results = {}
        model_result = {}
        
        for model_name, model in tqdm(zip(self.model_names, self.models)):
            # PlasmaProteinBinding and TetrahymenaPyriformisToxicity are regression models
            if model_name == 'PlasmaProteinBinding':
                model_result[model_name] = model.predict(features)[0].round(4).astype(str) +' %'
            
            elif model_name == 'TetrahymenaPyriformisToxicity': 
                model_result[model_name] = model.predict(features)[0].round(4).astype(str) +' pIGC50 (ug/L)'
            
            else:
                # First index ->  probability that the data belong to class 0,
                # Second index ->  probability that the data belong to class 1.
                model_result[model_name]  = model.predict_proba(features)[0][1].round(2)
        
        final_results[smiles] = model_result
        print(final_results)
        with open('final_results.json', 'w') as json_file:
            json.dump(str(final_results), json_file)
        print('Result file is saved')

if __name__=='__main__':
    
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser(description = "Predict ADMET properties using  openDMPK")
    ap.add_argument("-s", "--smiles", action = 'store', dest = 'smiles', 
            type = str, required = True, help = "SMILES string")
    
    args = vars(ap.parse_args())
    input_smiles = args['smiles']
    
    opendmpk = openDMPK()
    opendmpk.predict(input_smiles)
