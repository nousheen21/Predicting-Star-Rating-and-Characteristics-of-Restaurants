# Support Vector Machine (SVM)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('trainingdatamodified.csv')
dataset1 = pd.read_csv('testingdatamodified.csv')

X = dataset.iloc[:, 1:-3].values
#y = dataset.iloc[:, [672,673,674,675,676]].values
y = dataset.iloc[:, 676].values

X_test= dataset1.iloc[:, :-3].values
#y1_test= dataset1.iloc[:, [671,672,673,674,675]].values
y1_test= dataset1.iloc[:, 675].values
# Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X, y)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y1_test, y_pred)

from sklearn.metrics import accuracy_score

print('Accuracy Score:' ,accuracy_score(y1_test, y_pred))


import seaborn as sns   

ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt='g'); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix For SVM'); 
ax.xaxis.set_ticklabels(['True', 'False']); ax.yaxis.set_ticklabels(['True', 'False']);
