 ##### Book renting project

import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import xgboost

######################################   Importing all Data ################################################

############## Customer features #######################
#Importing Customer features data
cust_f = pd.read_csv("C:\\Users\\kode surendra aba\\Desktop\\Data science\\python\\Bookstartup_assignment\\data_sets\\customer_features.csv")

cust_f.head()

# Remove '-' pattern from favorite_genres column
cust_f['favorite_genres'] = cust_f.favorite_genres.apply(lambda x: x.replace("-", ""))

# Convert favorite_genres column to lowercase
cust_f['favorite_genres'] = cust_f.favorite_genres.apply(lambda x: x.lower())

#Create favorite_genres column to numeric dummy variables
vect = CountVectorizer()
X = vect.fit_transform(cust_f.favorite_genres)
cust_f = cust_f.join(pd.DataFrame(X.toarray(), columns=vect.get_feature_names()))

#Remove favorite_genres column since we have converted it to dummy variables
cust_f.drop('favorite_genres', inplace=True, axis=1)



#Importing Last Month Assprtment data
last_assort = pd.read_csv("C:\\Users\\kode surendra aba\\Desktop\\Data science\\python\\Bookstartup_assignment\\data_sets\\last_month_assortment.csv")

last_assort.head()

# since shipping cost is 0.60 paise per side per book
last_assort['shipping_cost'] = last_assort.purchased.apply(lambda x: 0.6 if x==True else 1.2)

#Importing Product features data
prod_f = pd.read_csv("C:\\Users\\kode surendra aba\\Desktop\\Data science\\python\\Bookstartup_assignment\\data_sets\\product_features.csv")

prod_f.head()

# Creating a master dataframe by merging the provided dataframes
master_df = pd.merge(prod_f, last_assort, on ='product_id')
master_df.head()


#Importing original purchase order data
og_po = pd.read_csv("C:\\Users\\kode surendra aba\\Desktop\\Data science\\python\\Bookstartup_assignment\\data_sets\\original_purchase_order.csv")



#Merging og_po data with master data
master_df = pd.merge(master_df, og_po, on ='product_id')
master_df.head()

#Merging customer data with master data
master_df = pd.merge(master_df, cust_f, on='customer_id')

master_df.columns

#Importing next month purchase order data
nxt_po = pd.read_csv("C:\\Users\\kode surendra aba\\Desktop\\Data science\\python\\Bookstartup_assignment\\data_sets\\next_purchase_order.csv", index_col=0)



#Calculate loan amount using user defined function calculate_loan

def calculate_loan(df_qty, df_cost):
    df_total_cost = df_qty*df_cost
    return round(sum(df_total_cost),2)

prev_month_cost = round(sum(master_df['shipping_cost']),2)
print('last month total shipping cost:', prev_month_cost)


prev_month_loan = calculate_loan(og_po['quantity_purchased'], og_po['cost_to_buy'])
print('last_month_loan: ', prev_month_loan)

next_month_cost = calculate_loan(nxt_po['quantity_purchased'], nxt_po['cost_to_buy'])
print('next_month_cost: ', next_month_cost)

#Func to get accuracy and confusion matrix of models
def generate_accuracy_and_heatmap(model, x, y):
    cm = confusion_matrix(y,model.predict(x))
    sns.heatmap(cm,annot=True, fmt="d")
    
    ac=accuracy_score(y, model.predict(x))
    f_score = f1_score(y, model.predict(x))
    print('Accuracy is: ', ac)
    print('F1 Score is: ', f_score)
    print('\n')
    print(pd.crosstab(pd.Series(model.predict(x), name='Predicted'),
                      pd.Series(y['purchased'], name='Actual')))
    return 1

#Collecting numeric columns together for correlation matrix
numeric_feature_columns = list(master_df._get_numeric_data().columns)
numeric_feature_columns

target='purchased'

k=15   # number of variables for heatmap
cols = master_df[numeric_feature_columns].corr().nlargest(k, target)[target].index
cm = master_df[cols].corr()
from matplotlib import figure
plt.figure( figsize =(15,12) )
sns.heatmap(cm, annot=True, cmap='viridis')
plt.show()

# Using label encoder to convert age_group to numerical code (or group)
le = LabelEncoder()
master_df['age_bucket'] = master_df['age_bucket'].astype(str)
master_df['age_bucket'] = le.fit_transform(master_df['age_bucket'])
master_df['fiction'] = le.fit_transform(master_df['fiction'])
master_df['genre'] = le.fit_transform(master_df['genre'])
master_df['is_returning_customer'] = le.fit_transform(master_df['is_returning_customer'])
master_df['purchased'] = le.fit_transform(master_df['purchased'])


#Model building
predictors = ['retail_value', 'length', 'difficulty','fiction', 'genre', 'age_bucket', 
                'is_returning_customer', 'beachread', 'biography', 'classic', 'drama', 'history', 
                'poppsychology', 'popsci', 'romance', 'scifi', 'selfhelp', 'thriller']
#Y = master_df['purchased']
X = master_df[predictors]


#X = master_df.loc[:, master_df.columns != target]
Y = master_df.loc[:, master_df.columns == target]

X.shape
Y.shape

x_train, x_test, y_train, y_test = train_test_split(X, Y,test_size=0.33,random_state=8)



clf_lr = LogisticRegression()
lr_baseline_model = clf_lr.fit(x_train,y_train)

generate_accuracy_and_heatmap(lr_baseline_model, x_test, y_test)

# Let's try GridSearchCV to improve things
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
svmmodel = SVC()
svmmodel.fit(x_train,y_train)
predictions = svmmodel.predict(x_test)

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


# Find the best 'C' value
param_grid = {'C': [0.1,1, 10, 100, 1000]}
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)
grid.fit(x_train,y_train)  
grid.best_params_
c_val = grid.best_estimator_.C

print('grid.best_estimator_.C : ', c_val)
print('Best Parameter Value: ', grid.best_params_)

#Best Parameter Value:  {'C': 1}



# Now we can re-run predictions on this grid object just like we would with a normal model.
grid_predictions = grid.predict(x_test)

# We will use the best 'C' value found by GridSearch and reload our LogisticRegression module
logmodel = LogisticRegression(C=c_val)
logmodel.fit(x_train,y_train)

print(confusion_matrix(y_test,grid_predictions))
print(classification_report(y_test,grid_predictions))

#              precision    recall  f1-score   support
#
#           0       0.79      0.83      0.81      7697
#           1       0.66      0.60      0.63      4183
#   micro avg       0.75      0.75      0.75     11880
#   macro avg       0.73      0.72      0.72     11880
#weighted avg       0.75      0.75      0.75     11880

# Now things are looking good!!



# We will use the best 'C' value found by GridSearch and reload our SVM module
svm_model = SVC(C=c_val)
svm_model.fit(x_train,y_train)

print(confusion_matrix(y_test,grid_predictions))
print(classification_report(y_test,grid_predictions))


y_pred = logmodel.predict(x_test)
print('Accuracy of Logistic Regression classifier on test set: {:.2f}'.format(metrics.accuracy_score(y_test, y_pred)))

y_pred = svm_model.predict(x_test)
print('Accuracy of Support vector Machine classifier on test set: {:.2f}'.format(metrics.accuracy_score(y_test, y_pred)))

############ XGBoost #################
from xgboost import XGBClassifier

xgb_model = XGBClassifier(C=c_val)
xgb_model.fit(x_train, y_train)

y_pred = xgb_model.predict(x_test)
print('Accuracy of adaboost classifier on test set: {:.2f}'.format(metrics.accuracy_score(y_test, y_pred)))

xgb_model = XGBClassifier(C=c_val)
xgb_model.fit(x_train, y_train)

generate_accuracy_and_heatmap(xgb_model, x_test, y_test)

#Accuracy is:  0.7611111111111111
#F1 Score is:  0.6436464088397791

temp_df = pd.DataFrame(last_assort.groupby(['product_id'])['purchased'].sum().reset_index())
temp_df.head()



prev_month_books_remaining = pd.merge(og_po, temp_df, on = 'product_id')
prev_month_books_remaining['quantity_remaining'] = prev_month_books_remaining['quantity_purchased'] - prev_month_books_remaining['purchased']
prev_month_books_remaining = prev_month_books_remaining.drop(columns = ['quantity_purchased', 'purchased'])
prev_month_books_remaining.head()


#Let's make predictions for next month's assortment:
next_assort = pd.read_csv("C:\\Users\\kode surendra aba\\Desktop\\Data science\\python\\Bookstartup_assignment\\data_sets\\next_month_assortment.csv")
next_assort.head()



prev_month_sales_df = pd.merge(og_po, last_assort, on ='product_id')
prev_month_sales_df.head()

total_sales_prev_month = round(sum(prev_month_sales_df['retail_value'].where(prev_month_sales_df['purchased']==True, 0)),2)
total_sales_prev_month  # 151617.36

next_month_pred = pd.merge(next_assort, prev_month_books_remaining, on = 'product_id')
next_month_pred = pd.merge(next_month_pred, cust_f, on = 'customer_id')
next_month_pred = pd.merge(next_month_pred, prod_f, on = 'product_id')
next_month_pred.head()


next_month_pred['age_bucket'] = next_month_pred['age_bucket'].astype(str)
next_month_pred['fiction'] = le.fit_transform(next_month_pred['fiction'])
next_month_pred['genre'] = le.fit_transform(next_month_pred['genre'])
next_month_pred['age_bucket'] = le.fit_transform(next_month_pred['age_bucket'])
next_month_pred['is_returning_customer'] = le.fit_transform(next_month_pred['is_returning_customer'])
next_month_pred.head()

#Selecting features
features_for_prediction = ['retail_value', 'length', 'difficulty','fiction', 'genre', 'age_bucket', 
                'is_returning_customer', 'beachread', 'biography', 'classic', 'drama', 'history', 
                'poppsychology', 'popsci', 'romance', 'scifi', 'selfhelp', 'thriller']
X = next_month_pred[features_for_prediction]
X.head()

print(X.shape)
print(len(next_month_pred['product_id'].unique()))


#Predict by xgb


predict_next_month_purchase = xgb_model.predict(X)
print('Number of books predicted to be purchased: ',sum(predict_next_month_purchase))
sum(predict_next_month_purchase)/X.shape[0]

#Number of books predicted to be purchased:  11301
#Out[44]: 0.3139166666666667


# Calculating the shipping cost for next month's prediction
next_month_shipping = (sum(predict_next_month_purchase)*0.6 + (X.shape[0]-sum(predict_next_month_purchase)*1.2))
print('Shipping cost predictions for next month\'s assortment: ', next_month_shipping)


#calculate sales for next month:
next_month_pred['next_month_purchase_predictions'] = predict_next_month_purchase
#data_next.head()
next_sales = round(sum(next_month_pred['retail_value'].where(next_month_pred['next_month_purchase_predictions']==1, 0)),2)
print("Sale prediction for next month: ", next_sales)

# Will we be able to pay back loan and afford next book purchase?
tot_cost = prev_month_loan + next_month_cost + shipping_cost + next_month_shipping
tot_sales = total_sales_prev_month + next_sales
print(" Total Profit/Loss (Profit will be Positive, Loss will be Negative) \n Formula = (Total Sales - Total Cost to Us) =", total_sales - total_cost_to_us)


Final_Decision = ('Yes' if (tot_sales - tot_cost > 0) else 'No')
print(Final_Decision)