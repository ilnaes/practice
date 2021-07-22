# %%
import myprelude
from myprelude.trans import *
from tidytuesday import TidyTuesday
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px

# %%
tt = TidyTuesday("2021-05-04")
# %%
df = tt.data["water"].assign(
    report_date=lambda x: pd.to_datetime(x["report_date"])
)
df
# %%
df.isna().mean(axis=0)
# %%
(
    df.select_dtypes("object").apply(
        lambda x: len(x.value_counts(dropna=False)), axis=0
    )
)

# %%
df["status_id"].value_counts()

# %%
(
    df.assign(country_name=fct_reorder(fct_lump(df["country_name"], 10))).pipe(
        (sns.countplot, "data"), y="country_name"
    )
)

# %%
(
    df.assign(water_source=fct_reorder(fct_lump(df["water_source"], 10))).pipe(
        (sns.countplot, "data"), y="water_source"
    )
)
# %%
(
    df.assign(water_tech=fct_reorder(fct_lump(df["water_tech"], 10))).pipe(
        (sns.countplot, "data"), y="water_tech"
    )
)


# %%
(
    (df["status"].value_counts(dropna=False) / len(df))
    .cumsum()
    .to_frame()
    .reset_index()
    .query("status < 0.9")
)
