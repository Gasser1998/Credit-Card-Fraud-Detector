# -------------------------------------- Misc. imports------------------------------------------
import pandas as pd                                   # Used for Processing data
import numpy as np                                    # Used for working with arrays
import matplotlib.pyplot as mplt                      # Used for Visualization
from termcolor import colored as tc                   # this library is used for text customization
import itertools                                      # Iterator Functions

# --------------------- Imports for the classification models------------------------

from sklearn.preprocessing import StandardScaler      # Used for data normalization (organizing data)
from sklearn.model_selection import train_test_split  # Data split
from sklearn.tree import DecisionTreeClassifier       # Decision tree algorithm
from sklearn.neighbors import KNeighborsClassifier    # KNN algorithm
from sklearn.linear_model import LogisticRegression   # Logistic regression algorithm
from sklearn.svm import SVC                           # SVM algorithm
from sklearn.ensemble import RandomForestClassifier   # Random forest tree algorithm

# ---------------------Imports to evaluate the classification models----------------------

from sklearn.metrics import confusion_matrix          # evaluation metric
from sklearn.metrics import accuracy_score            # evaluation metric


#-----------------------------------------------------------------------------------------------------------------------


# Import DATA

data = pd.read_csv('creditcard.csv')
data.drop('Time', axis = 'columns', inplace = True)
print(data.head())

# Data Processing and Exploratory Data analysis (Discovering patterns and outliers in the dataset)

cases = len(data)
nonfraud_count = len(data[data.Class == 0])
fraud_count = len(data[data.Class == 1])
fraud_percent = round((fraud_count / nonfraud_count * 100),2)

# Below Print Statements utilize termcolor module to improve visualization

print(tc('Case Count', attrs=['bold']))
print(tc('----------------------------------------', attrs=['bold']))
print(tc(f'Total number of cases are {"{:,}".format(cases)}', attrs=['bold']))
print(tc(f'Non-fraud Cases count: {"{:,}".format(nonfraud_count)}', attrs=['bold']))
print(tc(f'Fraudulent Cases count: {fraud_count}', attrs=['bold']))
print(tc(f'Percentage of fraudulent cases: {fraud_percent}', attrs=['bold']))
print(tc('----------------------------------------', attrs=['bold']))

# Statistics of the Data

nonfraud_cases = data[data.Class == 0]
fraud_cases = data[data.Class == 1]


print(tc('Case Amount Statistics', attrs=['bold']))
print(tc('----------------------------------------', attrs=['bold']))
print(tc('Non-Fraud Case Amount Statistics', attrs=['bold']))
print(round(nonfraud_cases.Amount.describe(), 2))
print(tc('----------------------------------------', attrs=['bold']))
print(tc('Fraudulent Case Amount Statistics', attrs=['bold']))
print(round(fraud_cases.Amount.describe(), 2))

#  Normalizing the data

sc = StandardScaler()
amount = data['Amount'].values
print(amount)
data['Amount'] = sc.fit_transform(amount.reshape(-1, 1))
print(tc(data['Amount'].head(10),attrs=['bold']))

# Data Splitting & Defining Variables

X = data.drop('Class', axis=1).values
Y = data['Class'].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
print(tc('X_train samples: ', attrs=['bold']), X_train[:1])
print(tc('X_test samples: ', attrs=['bold']), X_test[0:1])
print(tc('Y_train samples: ', attrs=['bold']), Y_train[0:20])
print(tc('Y_test samples: ', attrs=['bold']), Y_test[0:20])

# -----------------Modeling | Explanations of classification models linked next to each model--------------------------------

# 1 - Decision Tree | https://chirag-sehra.medium.com/decision-trees-explained-easily-28f23241248

tree_model = DecisionTreeClassifier(max_depth=4, criterion='entropy')
tree_model.fit(X_train,Y_train)
tree_yhat = tree_model.predict(X_test)


# 2 - K-Nearest Neighbors | https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761

n = 5

knn = KNeighborsClassifier(n_neighbors=n)
knn.fit(X_train, Y_train)
knn_yhat = knn.predict(X_test)
print(knn_yhat)

# # 3 - Logistic Regression | https://searchbusinessanalytics.techtarget.com/definition/logistic-regression

lr = LogisticRegression()
lr.fit(X_train, Y_train)
lr_yhat = lr.predict(X_test)


# # 4 SVM Support Vector Machines | https://www.kdnuggets.com/2016/07/support-vector-machines-simple-explanation.html

svm = SVC()
svm.fit(X_train, Y_train)
svm_yhat = svm.predict(X_test)
print(svm_yhat)

# # 5 Random Forest Tree | https://www.section.io/engineering-education/introduction-to-random-forest-in-machine-learning/

rf = RandomForestClassifier(max_depth=4)
rf.fit(X_train, Y_train)
rf_yhat = rf.predict(X_test)

# Testing each Model's accuracy

print(tc('ACCURACY SCORE', attrs = ['bold']))
print(tc('------------------------------------------------------------------------', attrs = ['bold']))
print(tc('Accuracy score of the Decision Tree model is {}'.format(accuracy_score(Y_test, tree_yhat)), attrs = ['bold']))
# print(tc('------------------------------------------------------------------------', attrs = ['bold']))
# print(tc('Accuracy score of the KNN model is {}'.format(accuracy_score(Y_test, knn_yhat)), attrs = ['bold'], color = 'green'))
# print(tc('------------------------------------------------------------------------', attrs = ['bold']))
# print(tc('Accuracy score of the Logistic Regression model is {}'.format(accuracy_score(Y_test, lr_yhat)), attrs = ['bold'], color = 'red'))
# print(tc('------------------------------------------------------------------------', attrs = ['bold']))
# print(tc('Accuracy score of the SVM model is {}'.format(accuracy_score(Y_test, svm_yhat)), attrs = ['bold']))
# print(tc('------------------------------------------------------------------------', attrs = ['bold']))
# print(tc('Accuracy score of the Random Forest Tree model is {}'.format(accuracy_score(Y_test, rf_yhat)), attrs = ['bold']))
# print(tc('------------------------------------------------------------------------', attrs = ['bold']))

# 3. Confusion Matrix

# defining the plot function

def plot_confusion_matrix(cm, classes, title, normalize = False,cmap = mplt.cm.Blues):
    title = 'Confusion Matrix of {}'.format(title)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    mplt.imshow(cm, interpolation = 'nearest',cmap = cmap)
    mplt.title(title)
    mplt.colorbar()
    tick_marks = np.arange(len(classes))
    mplt.xticks(tick_marks, classes, rotation = 45)
    mplt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        mplt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment = 'center',
                 color = 'white' if cm[i, j] > thresh else 'black')

    mplt.tight_layout()
    mplt.ylabel('True label')
    mplt.xlabel('Predicted label')

# Compute confusion matrix for the models

tree_matrix = confusion_matrix(Y_test, tree_yhat, labels = [0, 1]) # Decision Tree
# knn_matrix = confusion_matrix(Y_test, knn_yhat, labels = [0, 1]) # K-Nearest Neighbors
# lr_matrix = confusion_matrix(Y_test, lr_yhat, labels = [0, 1]) # Logistic Regression
# svm_matrix = confusion_matrix(Y_test, svm_yhat, labels = [0, 1]) # Support Vector Machine
# rf_matrix = confusion_matrix(Y_test, rf_yhat, labels = [0, 1]) # Random Forest Tree


# Plot the confusion matrix

mplt.rcParams['figure.figsize'] = (6, 6)

# 1. Decision tree

tree_cm_plot = plot_confusion_matrix(tree_matrix, classes = ['Non-Default(0)','Default(1)'], normalize = False, title = 'Decision Tree')
mplt.savefig('tree_cm_plot.png')
mplt.show()

# 2. K-Nearest Neighbors

# knn_cm_plot = plot_confusion_matrix(knn_matrix, classes = ['Non-Default(0)','Default(1)'], normalize = False, title = 'KNN')
# mplt.savefig('knn_cm_plot.png')
# mplt.show()

# # 3. Logistic regression

# lr_cm_plot = plot_confusion_matrix(lr_matrix, classes = ['Non-Default(0)','Default(1)'], normalize = False, title = 'Logistic Regression')
# mplt.savefig('lr_cm_plot.png')
# mplt.show()

# # 4. Support Vector Machine

# svm_cm_plot = plot_confusion_matrix(svm_matrix, classes = ['Non-Default(0)','Default(1)'], normalize = False, title = 'SVM')
# mplt.savefig('svm_cm_plot.png')
# mplt.show()

# 5. Random forest tree

# rf_cm_plot = plot_confusion_matrix(rf_matrix, classes = ['Non-Default(0)','Default(1)'], normalize = False, title = 'Random Forest Tree')
# mplt.savefig('rf_cm_plot.png')
# mplt.show()
#



