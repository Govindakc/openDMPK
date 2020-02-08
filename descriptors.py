# ************************************************#
# RDkit Molecular descriptors generation script   #
# Govinda KC                                      #
# UTEP, Computational Science                     #
# ************************************************#

import math
from rdkit.Chem import AllChem, Descriptors
from rdkit import Chem
import sys
import pandas as pd
import numpy as np
import os
from rdkit.ML.Descriptors import MoleculeDescriptors

class Features_Generations:
    
    def __init__(self, smiles):
        
        self.smiles = smiles
        
    def molecule_descriptors(self):
        
        descriptors = []
        """
        Receives the SMILES which is used to generate molecular descriptors (200) and saves as numpy file
        
        Parameter
        ---------
        
        input smiles : str
            Compouds in the form of smiles are used
    
        return : np.array
            Descriptors are saved in the form of numpy files
        """
       
        try:
            
            calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
            mol = Chem.MolFromSmiles(self.smiles)
            ds = calc.CalcDescriptors(mol)
            ds = list(ds)
            max_value = max(ds)
            if max_value > 10**30:
                return None
            
            ds = np.asarray(ds)
            descriptors.append(ds)
        
        except:
            return None
    
        features = ( np.asarray((descriptors), dtype=object) )
        return features
