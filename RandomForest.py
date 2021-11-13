import pandas as pd
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

import timeit

start = timeit.default_timer()

#Reading the input file
print("The following instruction will ask you to enter learning files. If you want to keep them to default, simply press Enter (Three times)")
print("Enter the training, validation, and testing file in a sequential manner")
inputData = input()
validateData = input()
testData = input()

for d in [5000]:
    for c in [1800]:
        #Reading the input file
        #if(inputData == "\n"):
        print("Model will run for - ")
        print("\nFile - train_c"+str(c)+"_d"+str(d))
        #inputData = pd.read_csv(r"C:\Users\Friday\Desktop\Fall21\CS6375\Homework3\all_data\\train_c"+str(c)+"_d"+str(d)+".csv", header=None)
        #Reading the validation file
        #if(validateData == "\n"):
        #validateData = pd.read_csv(r"C:\\Users\\Friday\\Desktop\\Fall21\\CS6375\\Homework3\\all_data\\valid_c"+str(c)+"_d"+str(d)+".csv", header=None)
        #if(testData == "\n"):
        #testData = pd.read_csv(r"C:\\Users\\Friday\\Desktop\\Fall21\\CS6375\\Homework3\\all_data\\test_c"+str(c)+"_d"+str(d)+".csv", header = None)

         # Adding relative path while submitting
        inputData = pd.read_csv(r"all_data\\train_c"+str(c)+"_d"+str(d)+".csv", header = None)
        validateData = pd.read_csv(r"all_data\\valid_c"+str(c)+"_d"+str(d)+".csv", header = None)
        testData = pd.read_csv(r"all_data\\test_c"+str(c)+"_d"+str(d)+".csv", header = None)

        #Storing the training data in X_train
        X_train1 = inputData.iloc[:,:-1]
        #Storing the class variable in y_train
        y_train1 = inputData.iloc[:,-1]
        #Storing the validation set in X_test
        X_train2 = validateData.iloc[:,:-1]
        #Storing the class variable in y_test6
        y_train2 = validateData.iloc[:,-1]

        #X_test = validateData.iloc[:,:-1]
        #Storing the class variable in y_test6
        #y_test = validateData.iloc[:,-1]

        X_test = testData.iloc[:,:-1]
        y_test = testData.iloc[:,-1]

        X_train = pd.concat([X_train1, X_train2], axis = 0)
        y_train = pd.concat([y_train1, y_train2], axis = 0)

        params ={
            'n_estimators' : [15, 200, 500],
            'criterion' : ['gini', 'entropy'],
            'max_depth' : [None, 300],
            'max_features' : ['auto', 'sqrt', None],
            'max_leaf_nodes' : [c, None],
            'bootstrap' : [True, False],
            'max_samples' : [None, 0.4],
            'warm_start' : [True, False]
        }
        #Run DecisionTreeClassifier to determine the hypermeters
        #model = DecisionTreeClassifier(criterion= 'gini', max_features= None, max_leaf_nodes= 30, splitter= 'random', min_samples_leaf=1, min_samples_split=2)
        #clf = BaggingClassifier(base_estimator=model, bootstrap= True, bootstrap_features= True, max_features= 50, max_samples= 100, n_estimators= 50, warm_start= True)
        clf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=300, max_features='auto', max_leaf_nodes=1800, bootstrap=False, max_samples=None, warm_start=True)
        # Uncomment next four lines to run GridSearchCV
        #clf = RandomForestClassifier()
        #clf = RandomizedSearchCV(clf, param_distributions=params)
        clf.fit(X_train, y_train)
        #print(clf.best_params_)
        y_pred = clf.predict(X_test)
        print("Accuracy:",metrics.accuracy_score(y_test, y_pred), "\t F1 Score - ", metrics.f1_score(y_test, y_pred))

stop = timeit.default_timer()
print("Runtime is = ", stop-start)