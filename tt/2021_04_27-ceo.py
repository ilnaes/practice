# %%
import myprelude
import pandas as pd
from pandas.tseries import offsets
import plotly.express as px
import seaborn as sns
import numpy as np
from tidytuesday import TidyTuesday

# %%
tt = TidyTuesday("2021-04-27")
# %%
def clean_date(date):
    if type(date) == str:
        if int(date[:4]) > 2021:
            return pd.to_datetime(str(int(date[:4]) - 1000) + date[4:])
        else:
            return pd.to_datetime(date)
    return date


df = tt.data["departures"].assign(
    departure_code=lambda x: x["departure_code"].astype("category"),
    leftofc=lambda x: x["leftofc"].map(clean_date),
)
# %%
(
    df.value_counts("departure_code")
    .reset_index()
    .pipe((sns.barplot, "data"), x=0, y="departure_code")
)

# %%
(
    df.query("leftofc.notna()")
    .assign(month=lambda x: x["leftofc"].dt.date - pd.offsets.MonthBegin(0))
    .value_counts(["month", "ceo_dismissal"])
    .sort_index()
    .reset_index()
    .pipe((sns.lineplot, "data"), x="month", y=0, hue="ceo_dismissal")
)

# %%
(
    df.query("leftofc.notna()")
    .assign(month=lambda x: x["leftofc"].dt.month)
    .value_counts(["month", "ceo_dismissal"])
    .unstack()
    .assign(
        p0=lambda x: x[0.0] / x[0.0].sum(), p1=lambda x: x[1.0] / x[1.0].sum()
    )[["p0", "p1"]]
    .reset_index()
    .melt("month")
    .pipe((sns.lineplot, "data"), x="month", y="value", hue="ceo_dismissal")
)

# %%
(df.value_counts(["ceo_dismissal", "max_tenure_ceodb"]))


# %%
df.query("max_tenure_ceodb == 3")
# %%
df.query("still_there.notna() and fyear_gone.notna()")

# %%
import nltk
from nltk.corpus import stopwords

sw = set(stopwords.words("english")) | set(["company", "ceo", "executive"])

(
    df.query("ceo_dismissal.notna()")
    .assign(
        notes=df["notes"]
        .str.lower()
        .str.replace(r"[^A-Za-z\s]", "")
        .str.split()
    )
    .explode("notes")
    .query("notes not in @sw")[
        ["ceo_dismissal", "dismissal_dataset_id", "notes"]
    ]
    .value_counts(["notes", "ceo_dismissal"])
    .where(lambda x: x > 10)
    .dropna()
    .unstack()
    .fillna(0)
    .assign(
        total0=lambda x: x[0.0].sum(),
        total1=lambda x: x[1.0].sum(),
        odd0=lambda x: (x[0.0] + 1) / (x["total0"] - x[0.0] + 1),
        odd1=lambda x: (x[1.0] + 1) / (x["total1"] - x[1.0] + 1),
        lor=lambda x: np.log(x["odd0"] / x["odd1"]),
    )
    .sort_values("lor")
)
