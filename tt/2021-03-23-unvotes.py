# %%
from myprelude import *
from tidytuesday import TidyTuesday
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# %%
tt = TidyTuesday("2021-03-23")
# %%
votes = tt.data["unvotes"].assign(vote=lambda x: x["vote"] == "yes")
rcall = tt.data["roll_calls"]
issues = tt.data["issues"]
# %%
rcall["date"] = pd.to_datetime(rcall["date"])
rcall["year"] = rcall["date"].dt.year
# %%
issues["issue"].value_counts()
# %%
(
    rcall.assign(year=rcall["date"].dt.year)
    .join(issues.set_index("rcid"), on="rcid")
    .groupby(["year", "issue"], dropna="False")
    .apply(lambda x: len(x))
    .reset_index()
    .pipe(px.area, x="year", y=0, color="issue")
)


# %%
votes["country"].value_counts().pipe(sns.kdeplot)
# %%
(
    votes.join(rcall.set_index("rcid"), on="rcid")
    .groupby("year")
    .agg({"vote": "mean"})
    .plot()
)
# %%
(rcall.groupby("year").agg({"importantvote": "mean"}).plot())


# %%
(
    votes.join(rcall.set_index("rcid"), on="rcid")
    .join(issues.set_index("rcid"), on="rcid")
    .groupby(["issue", "country"], dropna=False)
    .agg({"vote": ["count", "mean"]})
    .loc["Human rights", "vote"]["count"]
    .pipe(sns.kdeplot)
)

# %%
(
    votes.join(rcall.set_index("rcid"), on="rcid")
    .join(issues.set_index("rcid"), on="rcid")
    .groupby(["issue", "country"])
    .agg({"vote": ["count", "mean"]})
    .loc["Human rights", "vote"]
    .query("count >= 400")
    .sort_values("mean")
)

# %%
(
    votes.join(rcall.set_index("rcid"), on="rcid")
    .join(issues.set_index("rcid"), on="rcid")
    .groupby("issue")
    .agg({"vote": "mean"})
)

# %%
votes_rcall = votes.join(rcall.set_index("rcid"), on="rcid")
votes_rcall
# %%
(
    votes_rcall.query('country == "United States"')
    .groupby("year")
    .agg({"vote": "mean"})
    .plot()
)

len(rcall)

# %%
sns.barplot(
    data=votes.query(
        'country in ["United States", "Canada", "China", "Russia", "France", "United Kingdom"]'
    )
    .join(issues.set_index("rcid"), on="rcid")
    .groupby(["issue", "country"])
    .agg({"vote": "mean"})
    .reset_index(),
    y="issue",
    x="vote",
    hue="country",
)

# %%

vote_matrix = (
    votes.join(rcall.set_index("rcid"), on="rcid")
    .assign(year=lambda x: x["date"].dt.year)
    .query("date > 2016")
    .groupby("country")
    .apply(lambda df: df.assign(n=len(df)))
    .reset_index(drop=True)
    .query("n > 10")[["country", "rcid", "vote"]]
    .pivot("country", "rcid", "vote")
    .fillna(False)
)

# %%
from sklearn.cluster import KMeans

km = KMeans(n_clusters=3)

from myprelude.datasets import ls_datasets, load_dataset

codes = load_dataset("country_codes")
(
    pd.DataFrame(km.fit(vote_matrix).labels_, index=vote_matrix.index)
    .join(codes.set_index("name"), rsuffix="")
    .pipe(px.choropleth, locations="alpha-3", color=0)
)
# %%
from spacy.lang.en.stop_words import STOP_WORDS

punct = '[-\d\(\)&/:;",\.\$]'

drop_list = (
    rcall.assign(
        descr=lambda x: x["descr"]
        .str.lower()
        .str.replace(punct, "", regex=True)
        .str.split()
    )
    .explode("descr")
    .value_counts("descr")
    .drop(STOP_WORDS, axis=0, errors="ignore")[lambda x: (x < 62) | (x > 1000)]
    .index.tolist()
)

# %%

(
    votes_rcall.query('country == "United States"')
    .assign(
        descr=lambda x: x["descr"]
        .str.lower()
        .str.replace(punct, "", regex=True)
        .str.split()
    )
    .explode("descr")
    .groupby("descr")
    .agg({"vote": ["mean", "count"]})
    .drop(list(STOP_WORDS) + drop_list, errors="ignore")
    .sort_values(("vote", "mean"))
)

# %%
