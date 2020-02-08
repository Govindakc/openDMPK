# openDMPK
### Prediction of ADMET properties for drug or durg like compounds

## Set up
```bash
pip install -r requirements.txt
```
Usage: run_openDMPK.py [-h] [--smiles SMILES]


Example: 
        ```python run_openDMPK.py --smiles "OC(O)C(Cl)(Cl)Cl"
        ```    
Results: 
```
         {'OC(O)C(Cl)(Cl)Cl': {'AmesMutagenesis': 0.0, 'AvianToxicity': 0.07, 
                        'BBBpenetration': 0.08, 'Biodegradation': 0.37, 
                        'CYP2c9': 0.14, 'CYP2d6': 0.09, 'CaCO2': 0.46, 
                        'EyeCorrosion': 0.27, 'EyeIrritation': 0.83, 
                        'HumanIntestineAbsorption': 0.97, 'HumanOralBioavailability': 0.47, 
                        'OrganicCationTransporter2': 0.16, 'hERGG': 0.68, 
                        'PlasmaProteinBinding': '0.3422 %', 
                        'TetrahymenaPyriformisToxicity': '0.1885 pIGC50 (ug/L)'}}
```
