#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


# In[2]:


data=pd.read_csv("titanic-passengers.csv",sep=";")
data.head()


# In[3]:


encoder=LabelEncoder()
one_hot=pd.get_dummies(data['Sex'])
data=data.drop('Sex',axis=1)
data=data.join(one_hot)
data['Embarked']=encoder.fit_transform(data['Embarked'])
data['Age'].fillna(data['Age'].mean(),inplace=True)


# In[4]:


data.drop(['Ticket','Cabin','Fare','PassengerId','Name'],axis=1,inplace=True)


# In[5]:


data.head()


# In[6]:


data['Survived'].value_counts()


# In[7]:


data["Survived"]=data["Survived"].map({"Yes": 1, "No": 0})


# In[8]:


x = data[['Pclass', 'Age','SibSp','Parch','Embarked','female','male']]
y = data['Survived']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)  

logreg = LogisticRegression()   
logreg.fit(x_train, y_train)  
y_pred  = logreg.predict(x_test)    
print('Acuuracy=',accuracy_score(y_pred,y_test))


# In[9]:


confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True)


# In[10]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[11]:


precision = trueP / trueP + false P
recall = trueP / trueP + false N
f1-score = 2*trueP/(2*trueP+ falseP +falseN)


# In[ ]:





# In[14]:


# prédire sur le jeu de test avec le modèle optimisé
y_test_pred_cv = grid.decision_function(X_test_std)

# construire la courbe ROC du modèle optimisé
fpr_cv, tpr_cv, thr_cv = metrics.roc_curve(y_test, y_test_pred_cv)

# calculer l'aire sous la courbe ROC du modèle optimisé
auc_cv = metrics.auc(fpr_cv, tpr_cv)

# créer une figure
fig = plt.figure(figsize=(6, 6))

# afficher la courbe ROC précédente
plt.plot(fpr, tpr, '-', lw=2, label='gamma=0.01, AUC=%.2f' % auc)

# afficher la courbe ROC du modèle optimisé
plt.plot(fpr_cv, tpr_cv, '-', lw=2, label='gamma=%.1e, AUC=%.2f' %          (grid.best_params_['gamma'], auc_cv))
         

# donner un titre aux axes et au graphique
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('SVM ROC Curve', fontsize=16)

# afficher la légende
plt.legend(loc="lower right", fontsize=14)

# afficher l'image
plt.show()


# In[ ]:




