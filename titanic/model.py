# %%
from myprelude import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px


# %%
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
y_train = train["Survived"]
X_train = train.drop(["Survived"], axis=1)
# %%
from sklearn.preprocessing import FunctionTransformer


def impute_age(df):
    means = df.groupby("Pclass").agg({"Age": "mean"})["Age"].to_dict()
    return df[["Age"]].mask(df["Age"].isna(), df["Pclass"].map(means), axis=0)


age_imputer = FunctionTransformer(impute_age)
age_imputer.fit_transform(X_train)
# %%
# One hot encode obvious cateogires
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

simple_cats = ColumnTransformer(
    [
        (
            "OneHot",
            OneHotEncoder(handle_unknown="ignore"),
            ["Pclass", "Sex", "Embarked"],
        )
    ]
)
simple_cats.fit_transform(train).shape
# %%
# Passthrough obvious numeric columns
nums = ColumnTransformer(
    [("nums", "passthrough", ["Age", "SibSp", "Parch", "Fare"])]
)
# %%
# One hot encode cabin after preprocessing
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

cabin_cats = make_pipeline(
    FunctionTransformer(lambda x: pd.DataFrame(x["Cabin"].str.slice(stop=1))),
    SimpleImputer(
        missing_values=np.nan, strategy="constant", fill_value="None"
    ),
    OneHotEncoder(handle_unknown="ignore"),
)

# %%
from sklearn.pipeline import FeatureUnion

processor = FeatureUnion(
    [
        ("age", age_imputer),
        ("cabin", cabin_cats),
        ("simple_cats", simple_cats),
        ("nums", nums),
        ("missing", quick_miss(["Age"])),
    ]
)


# %%
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

xgb_params = {
    # "model__objective": ["binary:logistic", "binary:hinge"],
    "model__objective": ["binary:logistic"],
    # "model__learning_rate": [0.01, 0.03, 0.05, 0.07, 0.09, 0.1],
    "model__learning_rate": [0.03],
    # "model__n_estimators": [200, 400, 600, 800, 1000],
    "model__n_estimators": [1000],
    "model__max_depth": [3, 4, 5, 6, 7, 8],
    "model__min_child_weight": [0.5, 1, 1.5],
    "model__gamma": [0],
    "model__subsample": [0.5],
    "model__colsample_bytree": [0.5],
}

final = Pipeline([("processor", processor), ("model", xgb.XGBClassifier()),])
search = GridSearchCV(
    estimator=final, param_grid=xgb_params, cv=5, scoring="accuracy"
)
search.fit(X_train, y_train)

print(search.cv_results_)
print("\n")
print(search.best_params_)

# %%
px.line(
    x=xgb_params["model__n_estimators"],
    y=search.cv_results_["mean_test_score"],
)
# %%
x = []
y = []
c = []
for (p, s) in zip(
    search.cv_results_["params"], search.cv_results_["mean_test_score"],
):
    x.append(p["model__min_child_weight"])
    y.append(p["model__max_depth"])
    c.append(s)

px.scatter_3d(x=x, y=y, z=c, color=c)
# %%
preds = search.best_estimator_.predict(test)
pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": preds}).to_csv(
    "submission.csv", index=False
)
# %%
from sklearn.model_selection import ShuffleSplit, cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
# cross_val_score(final, X_train, y_train, cv=cv)
for train_index, test_index in cv.split(X_train):
    final.fit(X_train.iloc[train_index], y_train.iloc[train_index])


# %%
