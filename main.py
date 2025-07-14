import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree  import DecisionTreeRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score
import joblib


# 1. Load Data
df=pd.read_csv("housing.csv")

# 2. Creating a test set
df["income_cat"]=pd.cut(df["median_income"],bins=[0.0,1.5,3.0,4.5,6.0,np.inf],labels=[1,2,3,4,5])

split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

for train_index,test_index in split.split(df,df["income_cat"]):
    train_set=df.loc[train_index].drop("income_cat",axis=1)
    test_set=df.loc[test_index].drop("income_cat",axis=1)

# 3. we will work on the copy of data
housing=train_set.copy()

# 4. Separate features and labels
housing_labels=housing["median_house_value"].copy()
housing_features=housing.drop("median_house_value",axis=1)

# 5. Separate numerical and categorical columns
num_attrib=housing_features.drop("ocean_proximity",axis=1).columns.tolist()
cat_attribs=["ocean_proximity"]

# 6. Pipelines for numerical columns

num_pipeline=Pipeline([
    ("imputer",SimpleImputer(strategy="median")),
    ("scaler",StandardScaler())
])

# 7. Pipelines for categorial columns

cat_pipeline=Pipeline([
    ("onehot",OneHotEncoder(handle_unknown="ignore"))
])

# 8. Construct the full pipeline

full_pipeline=ColumnTransformer([
    ("num",num_pipeline,num_attrib),
    ("cat",cat_pipeline,cat_attribs)
])

# housing_prepaired=full_pipeline.fit_transform(housing_features)

# 9. Creating Pipeline for model

ran_reg=Pipeline([
    ("preprocess",full_pipeline),
    ("model",RandomForestRegressor())
])

lin_reg=Pipeline([
    ("preprocess",full_pipeline),
    ("model",LinearRegression())
])

dec_tree=Pipeline([
    ("preprocess",full_pipeline),
    ("model",DecisionTreeRegressor())
])


print("training the model...........")
# 10. Train Our Model

ran_reg.fit(housing_features,housing_labels)
ran_pred=ran_reg.predict(housing_features)
print("Random Forest Classifier")
ran_rmses=-cross_val_score(ran_reg,housing_features,housing_labels,scoring="neg_root_mean_squared_error",cv=10)
print(pd.Series(ran_rmses).describe())


lin_reg.fit(housing_features,housing_labels)
lin_pred=lin_reg.predict(housing_features)
print("Linear Regression")
lin_rmses=-cross_val_score(lin_reg,housing_features,housing_labels,scoring="neg_root_mean_squared_error",cv=10)
print(pd.Series(lin_rmses).describe())


dec_tree.fit(housing_features,housing_labels)
dec_pred=dec_tree.predict(housing_features)
print("Decision Tree")
dec_rmses=-cross_val_score(dec_tree,housing_features,housing_labels,scoring="neg_root_mean_squared_error",cv=10)
print(pd.Series(dec_rmses).describe())



print("done")

# As Random Forest Classifier is Doing Well we Will save that model
# joblib.dump(ran_reg,"model.pkl")
# print("Model Is Saved")

# As model.pkl size is so big
import gzip
with gzip.open("model_compressed.pkl.gz", "wb") as f:
    joblib.dump(ran_reg, f)


test_set.to_csv("input.csv",index=False)
print("Test set Saved to input.csv")

