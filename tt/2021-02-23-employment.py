# %%
from myprelude import *
from tidytuesday import TidyTuesday
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import plotly as px
import dill


def save_session():
    dill.dump_session("2021-02-23.db")


# %%
import os

if "2021-02-23.db" in os.listdir():
    dill.load_session("2021-02-23.db")

# %%
tt = TidyTuesday("2021-02-23")
# %%
print(tt.readme)
# %%
employed = tt.data["employed"]
# %%
employed["race_gender"].unique()
# %%
employed["major_occupation"].unique()
# %%
employed["industry"].unique()
# %%
# industry_total are for industry, race_gender, and year
employed.groupby(["industry", "race_gender", "year"]).agg(
    {"industry_total": "var"}
).dropna().sum()

# %%
employed.isna().sum(axis=0)
# %%
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder()
ohe.fit_transform(
    employed.assign(race_gender=employed["race_gender"].astype("category"))[
        ["race_gender"]
    ]
)
