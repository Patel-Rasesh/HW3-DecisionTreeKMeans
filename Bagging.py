import pandas as pd
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import timeit

start = timeit.default_timer()

#Reading the input file
print("The following instruction will ask you to enter learning files. If you want to keep them to default, simply press Enter")
print("Enter the training, validation, and testing file in a sequential manner")
inputData = input()
validateData = input()
testData = input()

for d in [1000]:
    for c in [1500]:
        #Reading the input file
        #if(inputData == 'n'):
        print("Model will run for - ")
        print("\nFile - train_c"+str(c)+"_d"+str(d))
        #inputData = pd.read_csv(r"C:\Users\Friday\Desktop\Fall21\CS6375\Homework3\all_data\\train_c"+str(c)+"_d"+str(d)+".csv", header=None)
        #Reading the validation file
        #if(validateData == 'n'):
        #validateData = pd.read_csv(r"C:\\Users\\Friday\\Desktop\\Fall21\\CS6375\\Homework3\\all_data\\valid_c"+str(c)+"_d"+str(d)+".csv", header=None)
        #if(testData == 'n'):
        #testData = pd.read_csv(r"C:\\Users\\Friday\\Desktop\\Fall21\\CS6375\\Homework3\\all_data\\test_c"+str(c)+"_d"+str(d)+".csv", header=None)

        # Adding relative path while submitting
        inputData = pd.read_csv(r"all_data\\train_c"+str(c)+"_d"+str(d)+".csv", header = None)
        validateData = pd.read_csv(r"all_data\\valid_c"+str(c)+"_d"+str(d)+".csv", header = None)
        testData = pd.read_csv(r"all_data\\test_c"+str(c)+"_d"+str(d)+".csv", header = None)

        #Storing the training data in X_train
        X_train1 = inputData.iloc[:,:-1]
        #Storing the class variable in y_train
        y_train1 = inputData.iloc[:,-1]
        X_train2 = validateData.iloc[:,:-1]
        #Storing the class variable in y_train
        y_train2 = validateData.iloc[:,-1]
        #Storing the validation set in X_test
        X_test = testData.iloc[:,:-1]
        #Storing the class variable in y_test6
        y_test = testData.iloc[:,-1]

        # Mixing Training and validation files
        X_train = pd.concat([X_train1, X_train2], axis = 0)
        y_train = pd.concat([y_train1, y_train2], axis = 0)
        params ={
            'n_estimators' : [10, 50],
            'max_samples' : [2, d],
            'max_features' : [1, 50],
            'bootstrap' : [True, False],
            'bootstrap_features' : [True, False],
            'warm_start' : [True, False]
        }
        print("\nFile - train_c"+str(c)+"_d"+str(d))
        #Run DecisionTreeClassifier to determine the hypermeters
        model = DecisionTreeClassifier(criterion= 'entropy', max_features= None, max_leaf_nodes= 150, splitter= 'random', min_samples_leaf=1, min_samples_split=2)
        clf = BaggingClassifier(base_estimator=model, bootstrap= True, bootstrap_features= False, max_features= 50, max_samples= 1000, n_estimators= 50, warm_start= False)
        #clf = BaggingClassifier(base_estimator=model)
        #clf = GridSearchCV(clf, param_grid=params)
        clf.fit(X_train, y_train)
        #print(clf.best_params_)
        y_pred = clf.predict(X_test)
        print("Accuracy:",metrics.accuracy_score(y_test, y_pred), "\t F1 Score - ", metrics.f1_score(y_test, y_pred))

stop = timeit.default_timer()
print("Runtime is = ", stop-start)