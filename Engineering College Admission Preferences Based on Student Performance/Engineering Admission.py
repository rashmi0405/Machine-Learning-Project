#Data Perprocessing (Convert Raw data into Clean data)

#import the package
import warnings
warnings.filterwarnings("ignore",category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#Reading csv file
df=pd.read_csv(r"F:/Study/Tri-Semeter 4/ML-Project/DataSet/Master Data_2017.csv")

df.head(5)

df.shape



#Removing the columns 
df.drop(['Sr. no.', 'First Name', 'Middle Name', 'Last Name', 'User Name','Present Address', 'Permanent Address', 'Date Of Birth', 'Sem I', 'Sem II', 'Sem III', 'Sem IV', 'Sem V', 'Sem VI', 'Sem VII',
       'Sem VIII', 'Aggregate Percentage(BE/MCA)', 'ME Sem I', 'ME Sem II',
       'ME Sem III', 'ME Sem IV', 'ME Aggregate', 'Number Of ATKT Live', 'Number Of ATKT Dead','Year Down', 'PASSSING OUT BATCH', 'Placement Company', 'Unnamed: 36',
       'Total Eligible Company', 'Total Applied Company'], axis=1, inplace=True)

df.columns

df.head()

df.isnull().sum()


#Changing cpga into %
def ChangeDType(value):
    if (value < 10) and (value > 0):
        return float(value*9.5)
    else:
        return value
df["SSC Percentage"] = df["SSC Percentage"].apply(ChangeDType)


#Chahnge Dtype
def ChangeDType(value):
    if "%" in (str(value)):
        return float(str(value).replace("%",""))
    else:
        return value
df["Diploma Percentage"] = df["Diploma Percentage"].apply(ChangeDType)


#Changing Nationality
def ChangeNationality(value):
    if ("IND" or "HINDU") in (str(value)).upper():
        return "Indian"
    elif "NAN" in (str(value)).upper():
        return "Indian"
    else:
        return (str(value)).capitalize()
df["Nationality"] = df["Nationality"].apply(ChangeNationality)


df["Nationality"].isnull().sum()

df["Nationality"].unique()

#drop the Degree
df=df[df["Degree"]!="ME"]

df.drop(['Degree'], axis=1, inplace=True)

#Misiing Value
df.isnull().sum()

#To find all non-missing value
df.loc[df["HSC Percentage"].notna() & df["Diploma Percentage"].notna(), 'HSC Percentage'] = 0




#Filling the null value
df["Diploma Percentage"].fillna(0, inplace=True)
df["HSC Percentage"].fillna(0, inplace=True)


df["Branch"].fillna(df["Branch"].mode().iloc[0], inplace = True)
df["Category"].fillna(df["Category"].mode().iloc[0], inplace = True)

df = df[df["SSC Percentage"].notna()]
df.isnull().sum()
#Changing the dtype 
df.dtypes

df["Diploma Percentage"] = pd.to_numeric(df["Diploma Percentage"])

df.dtypes

# Import label encoder (Convert Category variable into numeric)
from sklearn import preprocessing 
  
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
  
# Encode labels in column 'in all'. 
df['Gender']= label_encoder.fit_transform(df['Gender']) 
  
df['Gender'].unique() 

df['Category']= label_encoder.fit_transform(df['Category']) 
  
df['Category'].unique() 

df['Nationality']= label_encoder.fit_transform(df['Nationality']) 
  
df['Nationality'].unique() 

df['Branch']= label_encoder.fit_transform(df['Branch']) 
  
df['Branch'].unique() 

print(df.head(5))


#%%

#X and Y

X = df[['Gender', 'Category','SSC Percentage','HSC Percentage','Diploma Percentage','Education Gap', 'Nationality','Branch']]
print("X:",X.shape)
y = df[['College Name']]
print("y:",y.shape)
y = y.values.ravel()

#%%

#Feature Selection

#ExtraTreeClassifier 
from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
print(feat_importances)
print(X.columns)
feat_importances.nlargest(7).plot(kind='barh')
plt.show()

#%%
#EDA
#Exploratory data analysis (EDA) is an approach to analyzing data sets to summarize their main characteristics, often with visual methods. 

#Collegewise Count group by Gender
fig, ax = plt.subplots(figsize=(16,10))
sns.countplot(x = "College Name", hue="Gender" , data = df, ax=ax)

#Branchwise Count group by Gender
fig, ax = plt.subplots(figsize=(16,10))
sns.countplot(x = "Branch", hue="Gender" , data = df, ax=ax)

#Collegewise count group by category
fig, ax = plt.subplots(figsize=(16,10))
sns.countplot(x = "College Name", hue="Category" , data = df, ax=ax)

#Branchwise count group by Education Gap
fig, ax = plt.subplots(figsize=(16,10))
sns.countplot(x = "Branch", hue="Education Gap" , data = df, ax=ax)


#Categorywise count
df.groupby("Category")["Category"].count()/df.shape[0]*100


#%%

#Splitting the data
from sklearn.model_selection import train_test_split
test_size=1/3
seed=42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = seed) 

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)





#%%
#Applying the model
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

# training a linear SVM classifier 
#It uses a technique called the kernel trick to transform your data and then based on these transformations it finds an optimal boundary between the possible outputs
from sklearn.svm import SVC 

svm_model_linear = SVC(kernel = 'linear', C = 1)
svm_model_linear.fit(X_train, y_train) 

y_train_pred=svm_model_linear.predict(X_train)
print('\n\nTrain Results...     Accuracy : ',accuracy_score(y_train,y_train_pred))

y_test_pred=svm_model_linear.predict(X_test)
print('\nTest Results...      Accuracy : ',accuracy_score(y_test,y_test_pred))

print('confusion marix :')
print(confusion_matrix(y_test,y_test_pred))
print('\n\nClassification Reports :')
print(classification_report(y_test,y_test_pred))

print('\n\nConfusion matrix using Crosstab :')
print(pd.crosstab(y_test,y_test_pred,margins=True,rownames=['Actual'],colnames=['Prediction']))

#%%
#Decision Tree
#The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features

from sklearn.tree import DecisionTreeClassifier

clf=DecisionTreeClassifier(criterion='entropy',max_depth=6,random_state=seed)
clf.fit(X_train,y_train)

print('\n\nFeature Importances :', clf.feature_importances_)

y_train_pred=clf.predict(X_train)
print('\n\nTrain Results...     Accuracy : ',accuracy_score(y_train,y_train_pred))

y_test_pred=clf.predict(X_test)
print('\nTest Results...      Accuracy : ',accuracy_score(y_test,y_test_pred))

print('confusion marix :')
print(confusion_matrix(y_test,y_test_pred))
print('\n\nClassification Reports :')
print(classification_report(y_test,y_test_pred))


print('\n\nConfusion matrix using Crosstab :')
print(pd.crosstab(y_test,y_test_pred,margins=True,rownames=['Actual'],colnames=['Prediction']))
#%%
#Random Forest Tree
#Random forests creates decision trees on randomly selected data samples, gets prediction from each tree and selects the best solution by means of voting.
from sklearn.ensemble import RandomForestClassifier

num_trees = 50
n_jobs = -1 

rd = RandomForestClassifier(n_estimators=num_trees,max_leaf_nodes=15,n_jobs=n_jobs,random_state=seed)
rd.fit(X_train, y_train)
y_pred=rd.predict(X_test)
print('\n\nFeature Importances :', rd.feature_importances_)

y_train_pred=rd.predict(X_train)
print('\n\nTrain Results...     Accuracy : ',accuracy_score(y_train,y_train_pred))

y_test_pred=rd.predict(X_test)
print('\nTest Results...      Accuracy : ',accuracy_score(y_test,y_test_pred))

print('\n\nConfusion matrix using Crosstab :')
print(pd.crosstab(y_test,y_test_pred,margins=True,rownames=['Actual'],colnames=['Prediction']))

#%%
#Navie Bayes
#It predicts membership probabilities for each class such as the probability that given record or data point belongs to a particular class. 
#It's also assumed that all the features are following a gaussian distribution i.e, normal distribution.

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB

num_tree=100
num_folds=10
lr=0.1


kfold=StratifiedKFold(n_splits=num_folds,random_state=seed)

gnb_clf=GaussianNB()
gnb_clf.fit(X_train,y_train)
results=cross_val_score(gnb_clf,X_train,y_train,cv=kfold)
print("CV-Accuracy:" ,results.mean())


y_train_pred=gnb_clf.predict(X_train)
print("Train------ Accuracy:",accuracy_score(y_train,y_train_pred))


y_test_pred=gnb_clf.predict(X_test)
print("Test------ Accuracy:",accuracy_score(y_test,y_test_pred))


#%%

#KNN
#Its purpose is to use a database in which the data points are separated into several classes to predict the classification of a new sample point.
from sklearn.neighbors import KNeighborsClassifier

k=7
knn_clf=KNeighborsClassifier(n_neighbors=k)
knn_clf.fit(X_train,y_train)

results=cross_val_score(knn_clf,X_train,y_train,cv=kfold)
print("CV-Accuracy:" ,results.mean())


y_train_pred=knn_clf.predict(X_train)
print("Train------ Accuracy:",accuracy_score(y_train,y_train_pred))


y_test_pred=knn_clf.predict(X_test)
print("Test------ Accuracy:",accuracy_score(y_test,y_test_pred))

#%%
#Extra Tree Classifier

from sklearn.ensemble import ExtraTreesClassifier 

etc_clf = ExtraTreesClassifier(n_estimators=7,criterion= 'entropy',min_samples_split= 5,max_depth= 25, min_samples_leaf= 5)      
etc_clf.fit(X_train, y_train) 

y_train_pred=etc_clf.predict(X_train)
print('\n\nTrain Results...     Accuracy : ',accuracy_score(y_train,y_train_pred))

y_test_pred=etc_clf.predict(X_test)
print('\nTest Results...      Accuracy : ',accuracy_score(y_test,y_test_pred))

print('confusion marix :')
print(confusion_matrix(y_test,y_test_pred))
print('\n\nClassification Reports :')
print(classification_report(y_test,y_test_pred))

print('\n\nConfusion matrix using Crosstab :')
print(pd.crosstab(y_test,y_test_pred,margins=True,rownames=['Actual'],colnames=['Prediction']))

#%%
#Model Tuning(For the perfornamce)

#Adpative Boosting(it focuses on classification problems and aims to convert a set of weak classifiers into a strong one.)

from sklearn.ensemble import AdaBoostClassifier
dt_clf_boost=AdaBoostClassifier(DecisionTreeClassifier(max_depth=1.0,random_state=seed),n_estimators=num_tree,learning_rate=lr,random_state=seed)
dt_clf_boost.fit(X_train,y_train) 
results=cross_val_score(dt_clf_boost,X_train,y_train,cv=kfold)
print("\n DT (AdaBoost)---CV.Train :%.2f" % results.mean())
y_train_pred=dt_clf_boost.predict(X_train)
print("\n DT (AdaBoost)---Train :%.2f" % accuracy_score(y_train,y_train_pred))
y_test_pred=dt_clf_boost.predict(X_test)
print("\n DT (AdaBoost)---Test :%.2f" % accuracy_score(y_test,y_test_pred))

#%%
#XGBoost()
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

num_trees=100
num_folds=5

kfold=StratifiedKFold(n_splits=num_folds,random_state=seed)

xgb_clf=XGBClassifier(n_estimators=num_trees,objective='binary:logistic',seed=seed)
xgb_clf.fit(X_train,y_train,early_stopping_rounds=10,eval_set=[(X_test,y_test)],verbose=1)
results=cross_val_score(xgb_clf,X_train,y_train,cv=kfold)
print("\n XGBoost--CV.Train :%.2f" % results.mean())
y_train_pred=xgb_clf.predict(X_train)
print("\n XGBoost---Train :%.2f" % accuracy_score(y_train,y_train_pred))
y_test_pred=xgb_clf.predict(X_test)
print("\n XGBoost---Test :%.2f" % accuracy_score(y_test,y_test_pred))

xgb.plot_importance(xgb_clf)

#%%
#GridSearch(For hyperparameter tuning)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


pipeline=Pipeline([('knc', KNeighborsClassifier())])
parameters={'knc__n_neighbors':(1,3,5,7,9,11,13,15),'knc__weights':('uniform','distance')}

grid_search=GridSearchCV(estimator=pipeline,param_grid=parameters,n_jobs=1,cv=5,verbose=1,scoring='accuracy')
grid_search.fit(X_train,y_train)

print("Best Training Score: %0.3f" % grid_search.best_score_)

print("Best parameter set:")
best_parameters=grid_search.best_estimator_.get_params()
for param in sorted(parameters.keys()):
    print("\t %s: %r" %(param,best_parameters[param]))

y_test_pred=grid_search.predict(X_test)
print("Test------ Accuracy:",accuracy_score(y_test,y_test_pred))
#%%

