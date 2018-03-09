#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list = ['poi','salary'] # You will need to use more features
features_list = ['poi','salary', 'bonus', 'total_stock_value', 'from_to_poi'] 
#email_features = ['to_messages', 'email_address', 'from_poi_to_this_person', 
#                  'from_messages', 'from_this_person_to_poi']
				  
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop("TOTAL", 0)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
# Creating a feature sum from_poi_to_this_person + from_this_person_to_poi
for key, value in data_dict.iteritems():
    from_m = 0
    to_m = 0
    for k, v in value.iteritems():
	    if k=='from_poi_to_this_person':
		    if v=='NaN':
			    from_m = 0
		    else:
			    from_m = v
	    if k=='from_this_person_to_poi':
		    if v=='NaN':
			    to_m = 0
		    else:
		        to_m = v
    value['from_to_poi'] = from_m + to_m

my_dataset = data_dict
#quit()
#data_e.loc[:, ('from_to_poi')] = my_data.loc[:,('from_this_person_to_poi')] + my_data.loc[:,('from_poi_to_this_person')]

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

## Scaling data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 100))
scaler.fit(data)
data = scaler.transform(data)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn import tree
clf = tree.DecisionTreeClassifier(random_state = 0, min_samples_split = 6, splitter = 'random', min_samples_leaf = 2)
clf.fit(features, labels)
accuracy = clf.score(features, labels)
print "score: " + str(accuracy)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
	
pca = PCA(whiten=True, n_components=4).fit(features_train)
#print pca.explained_variance_ratio_
features_train_pca = pca.transform(features_train)
features_test_pca = pca.transform(features_test)

from sklearn.model_selection import GridSearchCV
parameters = {'max_depth':range(1, 10) ,'presort':[True, False] ,'criterion':['gini', 'entropy'], 'min_samples_split':range(2, 20), 'min_samples_leaf':range(1, 10), 'splitter':['best', 'random']}
clf = tree.DecisionTreeClassifier()
clf = GridSearchCV(clf, parameters)
#clf.fit(features_train_pca, labels_train)
#print clf.best_params_

#{'min_samples_split': 9, 'splitter': 'random', 'min_samples_leaf': 2}


clf = tree.DecisionTreeClassifier(presort=False, splitter = 'best', min_samples_leaf = 2, criterion= 'gini', min_samples_split = 2, max_depth=8)
clf.fit(features_train_pca, labels_train)
accuracy = clf.score(features_test_pca, labels_test)
print "score: " + str(accuracy)



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)