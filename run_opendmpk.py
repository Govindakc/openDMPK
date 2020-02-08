# ********************************* #
# Govinda KC                        #
# UTEP, Computational Science       #
# Last modified: 2/8/2020           #
# ********************************* #

# Usage: python run_opendmpk.py SMILES ( eg. python run_opendmpk.py "CC(O)CO" )

import os, sys, joblib, time, json
import pandas as pd
import numpy as np
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
            if model_name == 'PlasmaProteinBinding':
                model_result[model_name] = str(model.predict(features)[0].round(4)) +' %'
            
            elif model_name == 'TetrahymenaPyriformisToxicity': 
                model_result[model_name] = str(model.predict(features)[0].round(4)) +' pIGC50 (ug/L)'
            
            else:
                model_result[model_name]  = model.predict_proba(features)[0][1].round(2)
        
        final_results[smiles] = model_result
        print(final_results)
        with open('final_results.json', 'w') as json_file:
            json.dump(str(final_results), json_file)
        print('Result file is saved')

def main():
    smiles = sys.argv[1]
    opendmpk =openDMPK()
    opendmpk.predict(smiles)

if __name__=='__main__':
    main()
