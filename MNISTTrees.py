from sklearn.datasets import fetch_openml
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import timeit

start = timeit.default_timer()

X, y = fetch_openml('mnist_784', version = 1, return_X_y=True)

X = X/255
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

# Parameters for DecisionTree
# params = {
#     'splitter' : ['best', 'random'],
#     'criterion' : ['gini', 'entropy'],
#     'max_features' : ['auto', 'log2', 'sqrt', None],
#     'max_leaf_nodes' : [None, 50, 500]
# }

# Parameters for BaggingCLassifier
# params ={
#             'n_estimators' : [10, 50],
#             'max_samples' : [2, 300],
#             'max_features' : [1, 50],
#             'bootstrap' : [True, False],
#             'bootstrap_features' : [True, False],
#             'warm_start' : [True, False]
# }

# Parameters for RandomForestClassifier
# params ={
#             'n_estimators' : [15, 200, 500],
#             'criterion' : ['gini', 'entropy'],
#             'max_depth' : [None, 300],
#             'max_features' : ['auto', 'sqrt', None],
#             'max_leaf_nodes' : [500, None],
#             'bootstrap' : [True, False],
#             'max_samples' : [None, 0.4],
#             'warm_start' : [True, False]
# }

# Parameters for GradientBoostingClassifier
params ={
            'loss' : ['deviance', 'exponential'],
            'learning_rate' : [0.1, 0.5],
            'n_estimators' : [15, 100],
            'criterion' : ['friedman_mse', 'squared_error', 'mse', 'mae'],
            'max_depth' : [3, 10],
            'max_features' : ['auto', 'sqrt', None],
            'max_leaf_nodes' : [10, 300],
            'warm_start' : [True, False]
        }
# Tuning the classifer with RandomSearchCV
#model = DecisionTreeClassifier(splitter= 'best', max_leaf_nodes= None, max_features= None, criterion= 'entropy')
#clf = BaggingClassifier(base_estimator=model, warm_start= True, n_estimators= 50, max_samples= 300, max_features= 50, bootstrap_features=False, bootstrap= True)
clf = RandomForestClassifier(n_estimators=20, criterion='gini', max_depth=None, max_features='auto', max_leaf_nodes=200, bootstrap=False, max_samples=0.4, warm_start=True)
#clf = GradientBoostingClassifier(warm_start= False, n_estimators= 20, max_leaf_nodes= 200, max_features= 'auto', max_depth= 3, loss= 'deviance', learning_rate= 0.1, criterion= 'mse')
#clf = RandomizedSearchCV(clf, param_distributions=params)
clf.fit(X_train, y_train)
#print(clf.best_params_)
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))