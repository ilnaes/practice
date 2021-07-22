# %%
from myprelude import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import statsmodels.formula.api as smf
import statsmodels.api as sm

# %%
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
y_train = train["Survived"]
X_train = train.drop(["Survived"], axis=1)

print(f"test shape: {test.shape}")
print(f"train.shape: {train.shape}")

# %%
train
# %%
train.isna().sum()
# %%
train = train.assign(Cabin=train["Cabin"].str.slice(stop=1).fillna("None"))
# %%
def add_count(df):
    df["other"] = len(df) - 1
    return df


(
    train.groupby("Ticket")
    .apply(add_count)
    .assign(Cabin=train["Cabin"].str.slice(stop=1).fillna("None"))
    .query('Cabin != "None" and Age.isnull()', engine="python")
)

# %%
(
    train.assign(Cabin=fct_lump(train["Cabin"], 4)).pipe(
        (sns.kdeplot, "data"), x="Age", hue="Cabin", common_norm=False
    )
)
# %%
sm.stats.anova_lm(
    train.query("Age.notna()", engine="python")
    .pipe((smf.ols, "data"), "Age ~ C(Sex) * C(Pclass)")
    .fit()
)

# %%
# %%
