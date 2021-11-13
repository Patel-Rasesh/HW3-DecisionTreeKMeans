import pandas as pd
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn import metrics
import timeit
from sklearn.model_selection import GridSearchCV

start = timeit.default_timer()

#Reading the input file
print("The following instruction will ask you to enter learning files. If you want to keep them to default, simply press Enter")
print("Enter the training, validation, and testing file in a sequential manner")
inputData = input()
validateData = input()
testData = input()

# Hyperparameter setting for each of the training file
masterForLoop = [
    "criterion= 'gini', max_features= 'auto', max_leaf_nodes= 30, splitter= 'random', min_samples_leaf=0.5, min_samples_split=2",
    "criterion= 'entropy', max_features= 'sqrt', max_leaf_nodes= 500, splitter= 'random', min_samples_leaf=1, min_samples_split=0.5",
    "criterion= 'gini', max_features= None, max_leaf_nodes= 1000, splitter= 'random', min_samples_leaf=1, min_samples_split=2",
    "criterion= 'entropy', max_features= 'auto', max_leaf_nodes= 1500, splitter= 'random', min_samples_leaf=1, min_samples_split=2",
    "criterion= 'entropy', max_features= None, max_leaf_nodes= None, splitter= 'random', min_samples_leaf=1, min_samples_split=0.5",
    "criterion= 'gini', max_features= None, max_leaf_nodes= 30, splitter= 'random', min_samples_leaf=1, min_samples_split=2",
    "criterion= 'gini', max_features= None, max_leaf_nodes= 50, splitter= 'random', min_samples_leaf=1, min_samples_split=2",
    "criterion= 'gini', max_features= None, max_leaf_nodes= 100, splitter= 'random', min_samples_leaf=1, min_samples_split=2",
    "criterion= 'entropy', max_features= None, max_leaf_nodes= 150, splitter= 'random', min_samples_leaf=1, min_samples_split=2",
    "criterion= 'gini', max_features= None, max_leaf_nodes= 1800, splitter= 'best', min_samples_leaf=1, min_samples_split=0.4",
    "criterion= 'gini', max_features= None, max_leaf_nodes= 300, splitter= 'best', min_samples_leaf=1, min_samples_split=2",
    "criterion= 'gini', max_features= None, max_leaf_nodes= 500, splitter= 'best', min_samples_leaf=1, min_samples_split=2",
    "criterion= 'gini', max_features= None, max_leaf_nodes= 100, splitter= 'best', min_samples_leaf=1, min_samples_split=2",
    "criterion= 'gini', max_features= None, max_leaf_nodes= 150, splitter= 'random', min_samples_leaf=1, min_samples_split=2",
    "criterion= 'entropy', max_features= None, max_leaf_nodes= None, splitter= 'random', min_samples_leaf=1, min_samples_split=2"
]
# for d in [100, 1000, 5000]:
#     for c in [300, 500, 1000, 1500, 1800]:


for d in [1000]:
    for c in [1500]:
        #Reading the input file
        #if(inputData == '\n'):
        print("Model will run for - ")
        print("\nFile - train_c"+str(c)+"_d"+str(d))
        inputData = pd.read_csv(r"all_data\\train_c"+str(c)+"_d"+str(d)+".csv", header = None)
        validateData = pd.read_csv(r"all_data\\valid_c"+str(c)+"_d"+str(d)+".csv", header = None)
        testData = pd.read_csv(r"all_data\\test_c"+str(c)+"_d"+str(d)+".csv", header = None)
        #inputData = pd.read_csv(r"C:\Users\Friday\Desktop\Fall21\CS6375\Homework3\all_data\\train_c"+str(c)+"_d"+str(d)+".csv", header = None)
    #Reading the validation file
        #if(validateData == '\n'):
        #validateData = pd.read_csv(r"C:\\Users\\Friday\\Desktop\\Fall21\\CS6375\\Homework3\\all_data\\valid_c"+str(c)+"_d"+str(d)+".csv", header = None)
        #Reading the actual test file
        #if(testData == '\n'):
        #testData = pd.read_csv(r"C:\\Users\\Friday\\Desktop\\Fall21\\CS6375\\Homework3\\all_data\\test_c"+str(c)+"_d"+str(d)+".csv", header = None)
        
        #Storing the training data in X_train
        X_train1 = inputData.iloc[:,:-1]
        X_train2 = validateData.iloc[:,:-1]
        #Storing the class variable in y_train
        y_train1 = inputData.iloc[:,-1]
        y_train2 = validateData.iloc[:,-1]

        X_train = pd.concat([X_train1, X_train2], axis =0)
        y_train = pd.concat([y_train1, y_train2], axis = 0)
        #Storing the test set in X_test
        X_test = testData.iloc[:,:-1]
        #Storing the class variable in y_test
        y_test = testData.iloc[:,-1]

        #max_features = int(d/100)
        params = {'splitter' : ['best', 'random'],
                    'criterion' : ['gini', 'entropy'],
                    'max_features' : ['auto', 'log2', 'sqrt', None],
                    #'min_samples_leaf' : ['1', '10', '50', '100'],
                    #'min_samples_split' : ['2', '10'],
                    'max_leaf_nodes' : [None, c, int(c/10)]
                    } 
        # Uncomment the following four lines to perform GridSearchCV
        #model = DecisionTreeClassifier()
        #classifier = GridSearchCV(model, param_grid = params)
        #classifier.fit(X_train, y_train)
        #print(classifier.best_params_)

        #model = DecisionTreeClassifier(masterForLoop[i], min_samples_leaf=leaf, min_samples_split=split)
        model = DecisionTreeClassifier(criterion= 'entropy', max_features= None, max_leaf_nodes= 150, splitter= 'random', min_samples_leaf=1, min_samples_split=2)
        model = model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print("Accuracy:",metrics.accuracy_score(y_test, y_pred),"\tF1 score:", metrics.f1_score(y_test, y_pred))

stop = timeit.default_timer()
print("Runtime is = ", stop-start)