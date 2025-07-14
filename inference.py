import joblib 
import pandas as pd
from sklearn.metrics import r2_score
import gzip

housing=pd.read_csv("input.csv")

housing_labels=housing["median_house_value"].copy()
housing_features=housing.drop("median_house_value",axis=1)

with gzip.open('model_compressed.pkl.gz', 'rb') as f:
    model = joblib.load(f)

y_predict=model.predict(housing_features)

# new_df=pd.DataFrame(y_predict)

# df = pd.concat([housing, new_df], axis=1)
# df.to_csv("output.csv",index=False)

print(f"r2 score of model is {r2_score(housing_labels,y_predict)}")
