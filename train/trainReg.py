# ********************************************
# Script for Training (Regresson)       ******
# (Grid search for model model tuning)  ******
# ********************************************
#  Last modified: 4/29/2020

from hypopt import GridSearch
import os, sys, glob, joblib, json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
from sklearn import metrics
import numpy as np
from scipy import stats
from math import sqrt
import itertools

class Train_models: 
    def __init__(self, train=None,test=None,validation=None):
        self.train = train
        self.test = test
        self.val = validation
    
        self.random_state = 1
        
        self.reports = {}

    def split_data(self):
        
        if self.test is None and self.val is None:
            print('Using Combined train and test sests')
            x = self.train[:,:-1]
            y = self.train[:,-1]
            
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size = 0.25, \
                random_state = self.random_state)
            self.x_val = None
            self.y_val = None
            return True

        elif self.test is not None and self.val is None:
            print('Using separate train and test sets')
            
            self.x_train = self.train[:,:-1]
            self.y_train = self.train[:,-1]
            
            self.x_test = self.test[:,:-1]
            self.y_test = self.test[:,-1]
            
            self.x_val = None
            self.y_val = None
            return True
        
        else:
            print('Using separate train, test, and validation sets')

            self.x_train = self.train[:,:-1]
            self.y_train = self.train[:,-1]
            
            self.x_test = self.test[:,:-1]
            self.y_test = self.test[:,-1]

            self.x_val = self.val[:,:-1]
            self.y_val = self.val[:,-1]
            return True
    
    def rmse(self, y, f):
        rmse = sqrt(((y - f)**2).mean(axis=0))
        return rmse
    
    def write_results(self):
        # Save the report
        with open(os.path.join(REPORTS, dir_name, file_name+'.json'), 'w') as f:
            json.dump(self.reports, f)
            print('results saved') 

    def get_results(self, mdl):
        y_pre = mdl.predict(self.x_test)

        results = dict()
        results['r2_score'] = metrics.r2_score(self.y_test, y_pre)
        results['R'] = stats.pearsonr(self.y_test, y_pre)
        results['Rho'] = stats.spearmanr(self.y_test, y_pre)
        results['rmse'] = self.rmse(self.y_test, y_pre)
        
        if self.val is not None:
            results['data_info'] = {"train_count": len(self.x_train), \
                "test_count": len(self.x_test), "val_count":len(self.val)}
        else:
            results['data_info'] = {"train_count": len(self.x_train), "test_count": len(self.x_test)}
        
        return results
    
    def rf_regressor(self):
        
        clf = RandomForestRegressor(random_state = self.random_state)
        
        params = {'n_estimators': [int(x) for x in np.linspace(start=100, stop=800, num=8)],\
                        'max_features':['auto', 'sqrt', 'log2']}
        
        mdl = GridSearch(model=clf, param_grid=params)
        
        print('Fitting Rf')
        mdl.fit(self.x_train, self.y_train, self.x_val, self.y_val)
        self.reports['rf'] = self.get_results(mdl)
        
        model_path = os.path.join(MODELS, dir_name, file_name+'.rf')
        
        joblib.dump(mdl, model_path)
    
    def   xgb_regressor(self):
        
        clf = xgb.XGBRegressor(random_state=self.random_state)
        
        params = {
            'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3],
            'n_estimators': [int(x) for x in np.linspace(start=50, stop=800, num=16)],
            'colsample_bytree': [i/10.0 for i in range(3,11)]}

        mdl = GridSearch(model=clf, param_grid=params)

        print('Fitting XGBoost')
        mdl.fit(self.x_train, self.y_train)

        mdl.fit(self.x_train, self.y_train, self.x_val, self.y_val)

        self.reports['xgb'] = self.get_results(mdl)
                
        model_path = os.path.join(MODELS, dir_name, file_name+'.xgb')
        joblib.dump(mdl, model_path)

    def mlp_regressor(self):
        def get_hidden_layers():
            
            x = [64, 128, 256]
            hl = []

            for i in range(1, len(x)):
                hl.extend([p for p in itertools.product(x, repeat=i+1)])

            return hl
        
        clf = MLPRegressor(solver='adam', alpha=1e-5, early_stopping=True, \
                            random_state=self.random_state)
        
        hidden_layer_sizes = get_hidden_layers()
        params = {'hidden_layer_sizes': hidden_layer_sizes}

        mdl = GridSearch(model=clf, param_grid=params)
        
        mdl.fit(self.x_train, self.y_train, self.x_val, self.y_val)

        self.reports['mlp'] = self.get_results(mdl)
        
        model_path = os.path.join(MODELS, dir_name, file_name+'.mlp')
        joblib.dump(mdl, model_path)

    def train_all(self):
        
        self.rf_regressor()
        
        self.xgb_regressor()
        
        self.mlp_regressor()
       
        self.write_results()

if __name__=="__main__":

    MODELS = 'modelsReg'
    REPORTS = 'reportsReg'

    if not os.path.isdir(MODELS):
        os.mkdir(MODELS)
    if not os.path.isdir(REPORTS):
        os.mkdir(REPORTS)    
    
    file_name, _ = os.path.splitext(os.path.basename(sys.argv[1]))
    dir_name = os.path.dirname(sys.argv[1])
    print(dir_name)
    dir_name = dir_name.split('/')[3]

    print('directory name',dir_name)
    if not os.path.isdir(os.path.join(MODELS, dir_name)):
        os.mkdir(MODELS+'/'+dir_name)
        os.mkdir(REPORTS+'/'+dir_name)

    arguments = len(sys.argv) - 1

    if arguments ==3:
        train = np.load(sys.argv[1])
        test = np.load(sys.argv[2])
        valid = np.load(sys.argv[3])
        t = Train_models(train, test, valid)
    elif arguments == 2:
        train = np.load(sys.argv[1])
        test = np.load(sys.argv[2])
        valid = None
        t = Train_models(train, test, valid)
    else:
        train = np.load(sys.argv[1])
        test = None
        valid = None
        t = Train_models(train, test, valid)

    if not t.split_data():
        print('Features files are not present')
        exit()
    t.train_all()
