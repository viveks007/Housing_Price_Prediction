#!/usr/bin/env python
# coding: utf-8

# In[196]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[197]:


housing=pd.read_csv(r"D:\PC\Data Science\data sets\Housing\housing.csv")


# In[198]:


housing.head()


# In[199]:


housing.columns


# In[200]:


housing.info()


# In[201]:


housing["ocean_proximity"].value_counts()


# In[202]:


housing.describe()


# In[203]:


from sklearn.model_selection import train_test_split


# In[204]:


train_set, test_set= train_test_split(housing, test_size=0.2, random_state=42)


# In[205]:


housing["income_cat"]= pd.cut(housing["median_income"], 
                             bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                             labels=[1, 2, 3, 4, 5])


# In[206]:


housing_labels = train_set["median_house_value"].copy()


# In[207]:


housing["income_cat"].hist()


# In[208]:


housing.plot(kind="scatter", x="longitude", y="latitude")


# In[209]:


housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)


# In[210]:


housing.plot(kind="scatter", x="longitude", y="latitude",
            s=housing["population"]/100, label="population", figsize=(10,7),
            c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,)
plt.legend()


# In[211]:


corr_matrix= housing.corr()


# In[212]:


corr_matrix["median_house_value"].sort_values(ascending=False)


# In[213]:


from pandas.plotting import scatter_matrix


# In[214]:


attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]


# In[215]:


scatter_matrix(housing[attributes], figsize=(12,8))


# In[216]:


median= train_set["total_bedrooms"].median()


# In[217]:


train_set["total_bedrooms"].fillna(median, inplace=True)


# In[248]:


train_set.columns


# In[249]:


X_train_set=train_set.drop("median_house_value", axis=1)


# In[250]:


from sklearn.impute import SimpleImputer


# In[251]:


housing_num=X_train_set.drop("ocean_proximity", axis=1)


# In[252]:


imputer=SimpleImputer(strategy="median")


# In[253]:


imputer.fit(housing_num)


# In[254]:


imputer.statistics_


# In[255]:


housing_num.median().values


# In[256]:


X=imputer.transform(housing_num)


# In[257]:


housing_tr=pd.DataFrame(X, columns=housing_num.columns)


# In[258]:


housing_cat=train_set[["ocean_proximity"]]


# In[259]:


housing_cat.head()


# In[260]:


from sklearn.preprocessing import OrdinalEncoder


# In[261]:


ordinal_encoder=OrdinalEncoder()


# In[262]:


housing_cat_encoded= ordinal_encoder.fit_transform(housing_cat)


# In[263]:


housing_cat_encoded[:10]


# In[264]:


ordinal_encoder.categories_


# In[265]:


from sklearn.preprocessing import OneHotEncoder


# In[266]:


cat_encoder=OneHotEncoder()


# In[267]:


housing_cat_1hot= cat_encoder.fit_transform(housing_cat)


# In[268]:


housing_cat_1hot


# In[269]:


housing_cat_1hot.toarray()


# In[270]:


cat_encoder.categories_


# In[271]:


from sklearn.base import BaseEstimator, TransformerMixin


# In[272]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# In[273]:


num_pipeline= Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])


# In[274]:


housing_num_tr=num_pipeline.fit_transform(housing_num)


# In[275]:


from sklearn.compose import ColumnTransformer


# In[276]:


num_attribs= list(housing_num)
cat_attribs= ["ocean_proximity"]


# In[345]:


full_pipeline= ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])


# In[351]:


train_set.head()


# In[347]:


housing_prepared= full_pipeline.fit_transform(train_set)


# In[279]:


from sklearn.linear_model import LinearRegression


# In[280]:


lin_reg= LinearRegression()


# In[281]:


lin_reg.fit(housing_prepared, housing_labels)


# In[282]:


from sklearn.metrics import mean_squared_error


# In[283]:


housing_predictions= lin_reg.predict(housing_prepared)


# In[284]:


lin_mse= mean_squared_error(housing_labels, housing_predictions)


# In[285]:


lin_rmse= np.sqrt(lin_mse)


# In[286]:


lin_rmse


# In[287]:


from sklearn.metrics import r2_score


# In[288]:


r2_score(housing_labels, housing_predictions)


# In[289]:


from sklearn.tree import DecisionTreeRegressor


# In[290]:


tree_reg= DecisionTreeRegressor()


# In[291]:


tree_reg.fit(housing_prepared, housing_labels)


# In[292]:


housing_predictions= tree_reg.predict(housing_prepared)


# In[293]:


tree_mse=mean_squared_error(housing_labels, housing_predictions)


# In[294]:


tree_rmse=np.sqrt(tree_mse)


# In[295]:


tree_rmse


# In[296]:


from sklearn.model_selection import cross_val_score


# In[301]:


scores= cross_val_score(tree_reg, housing_prepared, housing_labels,
                        scoring="neg_mean_squared_error", cv=10)


# In[302]:


tree_rmse_scores= np.sqrt(-scores)


# In[303]:


def display_scores(scores):
    print("Scores", scores)
    print("Mean", scores.mean())
    print("SD", scores.std())


# In[304]:


display_scores(tree_rmse_scores)


# In[311]:


from sklearn.ensemble import RandomForestRegressor


# In[312]:


forest_reg= RandomForestRegressor()


# In[313]:


forest_reg.fit(housing_prepared, housing_labels)


# In[314]:


housing_predictions= forest_reg.predict(housing_prepared)


# In[315]:


forest_mse=mean_squared_error(housing_labels, housing_predictions)


# In[316]:


forest_rmse=np.sqrt(forest_mse)


# In[317]:


forest_rmse


# In[318]:


scores= cross_val_score(forest_reg, housing_prepared, housing_labels,
                        scoring="neg_mean_squared_error", cv=10)


# In[319]:


forest_rmse_scores= np.sqrt(-scores)


# In[320]:


display_scores(forest_rmse_scores)


# In[321]:


r2_score(housing_labels, housing_predictions)


# In[322]:


from sklearn.model_selection import GridSearchCV


# In[325]:


param_grid=[
    {'n_estimators':[3,10,30], 'max_features':[2,4,6,8]},
    {'bootstrap':[False], 'n_estimators':[3,10], 'max_features':[2,3,4]},
]


# In[326]:


forest_reg= RandomForestRegressor()


# In[327]:


grid_search=GridSearchCV(forest_reg, param_grid, cv=5,
                        scoring='neg_mean_squared_error',
                        return_train_score=True)


# In[328]:


grid_search.fit(housing_prepared, housing_labels)


# In[329]:


grid_search.best_params_


# In[330]:


grid_search.best_estimator_


# In[331]:


cvres= grid_search.cv_results_


# In[333]:


for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


# In[334]:


final_model= grid_search.best_estimator_


# In[360]:


X_test=test_set.drop("median_house_value", axis=1)


# In[342]:


y_test= test_set["median_house_value"].copy()


# In[354]:


test_set.head()


# In[361]:


test_set.columns


# In[363]:


X_test_prepared= full_pipeline.transform(test_set)


# In[364]:


final_predictions= final_model.predict(X_test_prepared)


# In[365]:


final_mse= mean_squared_error(y_test, final_predictions)


# In[366]:


final_rmse= np.sqrt(final_mse)


# In[367]:


final_rmse


# In[ ]:




