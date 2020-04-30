# *****************************************************************
# Script for different classifiers using Scikit-learn Library *****
# *****************************************************************

import numpy as np
import json, os, argparse
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, NuSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

#from sklearn.model_selection import train_test_split
#from sklearn.model_selection import StratifiedKFold,GridSearchCV
#import matplotlib.pyplot as plt
#from pandas.plotting import scatter_matrix

class Model_Development:
    def __init__(self, numpy_file, numpy_file2):
        self.numpy_file = numpy_file
        self.numpy_file2 = numpy_file2
        self.random_state = 123
        self.test_size = 0.25
        self.results = dict()
        
        self.output_dir = 'reports'
        if not os.path.isdir(self.output_dir): os.makedirs(self.output_dir)
        
        _, file_name = os.path.split(numpy_file2)
        self.file_name, _ = os.path.splitext(file_name)
        self.result_file = os.path.join(self.output_dir, f"{self.file_name}_results.json")        

    def evaluation_metrics(self, y_test, y_pred, y_prob):
        
        Scores = dict()
        
        roc_auc = metrics.roc_auc_score(y_test, y_prob)
        acc = metrics.accuracy_score(y_test, y_pred)
        
        f1_score = metrics.f1_score(y_test, y_pred, average = 'binary')
        kappa = metrics.cohen_kappa_score(y_test, y_pred)
        mcc = metrics.matthews_corrcoef(y_test, y_pred)
#        report = metrics.classification_report(y_test, y_pred)
        tp, fn, fp, tn = metrics.confusion_matrix(y_test, y_pred).ravel()
        SE = float(tp)/(tp+fn)
        SP = float(tn)/(tn+fp)
        PPV = float(tp)/(tp+fp)
        
        # Convert numpy.float64 to float using 'tolist()' and save the scores in dictonary 
        Scores['SE'] = SE.tolist()
        Scores['SP'] = SP.tolist()
        Scores['PPV'] = PPV.tolist()
        Scores['AUC'] = roc_auc.tolist()
        Scores['ACC'] = acc.tolist()
        Scores['F1_Score'] = f1_score.tolist()
        Scores['Cohen_Kappa'] = kappa.tolist()
        Scores['MCC'] = mcc#.tolist()
        Scores['TP'] = tp.tolist()
        Scores['TN'] = tn.tolist()
        Scores['FP'] = fp.tolist()
        Scores['FN'] = fn.tolist()
        
        return Scores
        
    def get_data_sets(self):
        
        train_data = np.load(self.numpy_file)
        test_data = np.load(self.numpy_file2)

        X_train = train_data[:,:-1]
        y_train = train_data[:, -1]
        X_test = test_data[:, :-1]
        y_test = test_data[:, -1]
        
        return X_train, X_test, y_train, y_test
    
    def write_results(self):
        with open(self.result_file, 'w') as f:
            json.dump(self.results, f)
            
    def random_forest_classifier(self):
        
        X_train, X_test, y_train, y_test = self.get_data_sets()
        
        model = RandomForestClassifier(random_state = self.random_state)
        print('Fitting model')
        model.fit(X_train, y_train) 
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]
    
        scores = self.evaluation_metrics(y_test, y_pred, y_prob)
        self.results['RF'] = scores
    
    def decision_tree_classifier(self):
        
        X_train, X_test, y_train, y_test = self.get_data_sets()
        
        model = DecisionTreeClassifier(random_state = self.random_state)
        print('Fitting model')
        model.fit(X_train, y_train) 
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]
        
        scores = self.evaluation_metrics(y_test, y_pred, y_prob)
        self.results['DT'] = scores
      
    def ada_boost_classifier(self):
        
        X_train, X_test, y_train, y_test = self.get_data_sets()
        
        model = AdaBoostClassifier(random_state = self.random_state)
        print('Fitting model')
        model.fit(X_train, y_train) 
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]
        
        scores = self.evaluation_metrics(y_test, y_pred, y_prob)
        self.results['ABA'] = scores  
    
    def logistic_regression(self):    
        
        X_train, X_test, y_train, y_test = self.get_data_sets()
        
        model = LogisticRegression()
        print('Fitting model')
        model.fit(X_train, y_train) 
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]
        
        scores = self.evaluation_metrics(y_test, y_pred, y_prob)
        self.results['LR'] = scores
    
    def xgb_classifier(self):

        X_train, X_test, y_train, y_test = self.get_data_sets()
        
        model = XGBClassifier(random_state = self.random_state)
        print('Fitting model')
        model.fit(X_train, y_train) 
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]
        
        scores = self.evaluation_metrics(y_test, y_pred, y_prob)
        self.results['XGB'] = scores
    
    def multinomial_nb(self):
        
        X_train, X_test, y_train, y_test = self.get_data_sets()
        
        model = MultinomialNB()
        print('Fitting model')
        model.fit(X_train, y_train) 
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]
        
        scores = self.evaluation_metrics(y_test, y_pred, y_prob)
        self.results['MNB'] = scores
    
    def gaussian_nb(self):
        
        X_train, X_test, y_train, y_test = self.get_data_sets()
        
        model = GaussianNB() 
        print('Fitting model')
        model.fit(X_train, y_train) 
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]
    
        scores = self.evaluation_metrics(y_test, y_pred, y_prob)
        self.results['GNB'] = scores
     
    def kneighbors_classifier(self):
        
        X_train, X_test, y_train, y_test = self.get_data_sets()
        
        model = KNeighborsClassifier() # random_state not found (not needed)
        print('Fitting model')
        model.fit(X_train, y_train) 
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]
        
        scores = self.evaluation_metrics(y_test, y_pred, y_prob)
        self.results['KNB'] = scores
    
    def dummy_classifier(self):
        
        X_train, X_test, y_train, y_test = self.get_data_sets()
        
        model = DummyClassifier(random_state = self.random_state)
        print('Fitting model')
        model.fit(X_train, y_train) 
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]
    
        scores = self.evaluation_metrics(y_test, y_pred, y_prob)
        self.results['DC'] = scores
    
    
    def mlp_classifier(self):
        
        X_train, X_test, y_train, y_test = self.get_data_sets()
        
        model = MLPClassifier(random_state = self.random_state)
        print('Fitting model')
        model.fit(X_train, y_train) 
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]

        scores = self.evaluation_metrics(y_test, y_pred, y_prob)
        self.results['MLP'] = scores

    
    def svc(self):
        
        X_train, X_test, y_train, y_test = self.get_data_sets()
         
        model = SVC(random_state = self.random_state, probability=True)
        print('Fitting model')
        model.fit(X_train, y_train) 
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]
    
        scores = self.evaluation_metrics(y_test, y_pred, y_prob)
        self.results['SVC'] = scores
    
    def nu_svc(self):
        
        X_train, X_test, y_train, y_test = self.get_data_sets()
        
        model = NuSVC(random_state = self.random_state, probability = True)
        print('Fitting model')
        model.fit(X_train, y_train) 
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]
    

        scores = self.evaluation_metrics(y_test, y_pred, y_prob)
        self.results['nuSVC'] = scores
    
    
    def gradient_boosting_classifier(self):
        
        X_train, X_test, y_train, y_test = self.get_data_sets()
        
        model = GradientBoostingClassifier(random_state = self.random_state)
        print('Fitting model')
        model.fit(X_train, y_train) 
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]
    
        scores = self.evaluation_metrics(y_test, y_pred, y_prob)
        self.results['GB'] = scores
    
    def linear_discriminant_analysis(self):

        X_train, X_test, y_train, y_test = self.get_data_sets()
        
        model = LinearDiscriminantAnalysis() # random_state not found
        print('Fitting model')
        model.fit(X_train, y_train) 
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]
    
        scores = self.evaluation_metrics(y_test, y_pred, y_prob)
        self.results['LDA'] = scores
    
    
    def quadratic_discriminant_analysis(self):
    
        X_train, X_test, y_train, y_test = self.get_data_sets()
        
        model = QuadraticDiscriminantAnalysis() # random_state not found
        print('Fitting model')
        model.fit(X_train, y_train) 
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]
    
        scores = self.evaluation_metrics(y_test, y_pred, y_prob)
        self.results['QDA'] = scores     
        
        
def main():
    md.random_forest_classifier()
    md.decision_tree_classifier()
    md.gradient_boosting_classifier()
    md.svc()
    md.xgb_classifier()
    md.mlp_classifier()
    md.quadratic_discriminant_analysis()
    md.linear_discriminant_analysis()
    md.dummy_classifier()
    md.kneighbors_classifier()
    md.logistic_regression()
    md.gaussian_nb()
   
    md.write_results()

if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description="Training models")
    
    parser.add_argument('--features', action='store', dest='features', required=True, \
                        help='features file (numpyFile)')
    parser.add_argument('--externalFeatures', action='store', dest = 'externalFeatures', required=True,\
                        help='externalFeatures file (numpyFile)')
    args = vars(parser.parse_args())
    
    numpy_file = args['features']
    numpy_file2 = args['externalFeatures']

    md = Model_Development(numpy_file, numpy_file2)
    main()
