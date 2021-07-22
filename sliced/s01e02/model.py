# %%
from myprelude import theil_u, seed_everything
from myprelude.trans import fct_lump
import pandas as pd
import numpy as np
import seaborn as sns
from seaborn.axisgrid import FacetGrid
import plotly.express as px

seed_everything(2021)

# %%
df = pd.read_csv("s01e02/train.csv", index_col="id")
test = pd.read_csv("s01e02/test.csv", index_col="id")
df.shape

# %%
from sklearn.model_selection import train_test_split

train, val = train_test_split(
    df, test_size=0.2, random_state=420, stratify=df["damaged"]
)

num_cols = [
    "incident_year",
    "incident_month",
    "incident_day",
    "height",
    "speed",
    "distance",
    "damaged",
]
fct_cols = [x for x in train.columns if x not in num_cols]
print(fct_cols)

# %%
sns.countplot(data=train, x="damaged")
print(train["damaged"].mean())
# 0.086 is minimum accuracy but logloss is real metric!
# HUGE class imbalance

# %%
train.isna().mean(axis=0)

# %%
train.nunique()

# %%
fct_lump(train[fct_cols].fillna("MISSING"), n=1000).apply(
    lambda x: theil_u(train["damaged"], x), axis=0
).to_frame().reset_index().pipe((sns.barplot, "data"), x=0, y="index")


# %%
(
    train.iloc[:, :-1]
    .isna()
    .join(train["damaged"])
    .corr()[["damaged"]]
    .reset_index()
    .melt(id_vars="index")
    .iloc[:-1]
    .dropna()
    .pipe((sns.barplot, "data"), x="value", y="index")
)

# %%
train[num_cols].corr().reset_index().melt(id_vars=["index"]).query(
    'index == "damaged"'
)[:-1].pipe((sns.barplot, "data"), x="value", y="variable")
# %%
train.pipe((sns.FacetGrid, "data"), col="damaged").map(
    sns.kdeplot, "incident_year"
)
# %%
train.pipe((sns.FacetGrid, "data"), sharey=False, col="damaged").map(
    sns.kdeplot, "height"
)
# %%
train.pipe((sns.FacetGrid, "data"), sharey=False, col="damaged").map(
    sns.kdeplot, "distance"
)
# %%
train.fillna("MISSING").pipe(
    (sns.FacetGrid, "data"), col="damaged", sharey=False
).map(sns.countplot, "aircraft_mass")

# %%
(
    fct_lump(train[["operator"]])
    .join(train["damaged"])
    .groupby(["operator"])
    .agg({"damaged": ["mean", "count"]})
    .loc[:, "damaged"]
    .reset_index()
    .pipe(px.bar, x="mean", y="operator", color="count")
)


# %%
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline, make_union
from myprelude.trans import PdWrap

cat_missing = make_union(
    make_pipeline(
        PdWrap(
            ColumnTransformer(
                [
                    (
                        "cat_missing",
                        SimpleImputer(
                            strategy="constant", fill_value="MISSING"
                        ),
                        fct_cols,
                    )
                ]
            ),
            cols=fct_cols,
        ),
        FunctionTransformer(lambda x: x.astype("category")),
    ),
    PdWrap(ColumnTransformer([("nums", "passthrough", num_cols)]), num_cols),
)

cat_missing.fit_transform(train)

# %%
