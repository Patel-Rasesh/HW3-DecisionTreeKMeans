import pandas as pd
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import timeit

start = timeit.default_timer()

#Reading the input file
#path = input()

for d in [100, 1000, 5000]:
    for c in [300, 500, 1000, 1500, 1800]:
        #Reading the input file
        inputData = pd.read_csv(r"C:\Users\Friday\Desktop\Fall21\CS6375\Homework3\all_data\\train_c"+str(c)+"_d"+str(d)+".csv")
        #Reading the validation file
        validateData = pd.read_csv(r"C:\\Users\\Friday\\Desktop\\Fall21\\CS6375\\Homework3\\all_data\\valid_c"+str(c)+"_d"+str(d)+".csv")

        #Storing the training data in X_train
        X_train = inputData.iloc[:,:-1]
        #Storing the class variable in y_train
        y_train = inputData.iloc[:,-1]
        #Storing the validation set in X_test
        X_test = validateData.iloc[:,:-1]
        #Storing the class variable in y_test
        y_test = validateData.iloc[:,-1]

        print("\nFile - train_c"+str(c)+"_d"+str(d))
        #Run DecisionTreeClassifier to determine the hypermeters
        for criterion in ['gini', 'entropy']:
        #for splitter in ['best', 'random']:
            model = DecisionTreeClassifier(criterion=criterion, min_samples_leaf=2)
            model = model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            print("For ",str(criterion))
            print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

            stop = timeit.default_timer()
        #print("Runtime is = ", stop-start)

        #Run gridSearchCV for min_sample_leaf