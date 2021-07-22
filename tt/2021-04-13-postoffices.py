# %%
import myprelude
from tidytuesday import TidyTuesday
import pandas as pd
import plotly.express as px
import seaborn as sns

# %%
tt = TidyTuesday("2021-04-13")
# %%
post = tt.data["post_offices"]
# %%
post.head()
# %%
post.shape

# %%
post["state"].value_counts(dropna=False)
# %%
(post.query("discontinued.isnull()")["state"].value_counts())

# %%
post.query("established.isnull()")
# %%
sorted(
    post.query("established.notnull()")[["established", "discontinued"]]
    .melt()
    .query("value.notnull()")
    .assign(variable=lambda x: x["variable"] == "established")["value"]
    .unique()
)

# %%
(
    post[["established", "discontinued"]]
    .query("established.notnull()")
    .melt()
    .query("value.notnull() and value > 1000 and value < 2010")
    .groupby(["value", "variable"])
    .size()
    .to_frame()
    .reset_index()
    .pipe(px.line, x="value", y=0, color="variable")
)

# %%
# count of number of post offices for every year
(
    post[["established", "discontinued"]]
    .query("established.notnull()")
    .melt()
    .query("value.notnull() and value > 1000 and value < 2010")
    .assign(variable=lambda x: 2 * (x["variable"] == "established") - 1)
    .groupby("value")
    .sum()
    .sort_index()
    .cumsum()
    .plot()
)
# %%
# distribution of ages of post offices
(
    post.query("established.notnull()")
    .assign(discontinued=post["discontinued"].fillna(2021))
    .assign(age=lambda x: x["discontinued"] - x["established"])
    .query("age >= 0 and age < 250")["age"]
    .hist(bins=25)
)

# %%
# distribution of establishment of short lived post offices
(
    post.query("established.notna() and discontinued.notna()")
    .assign(age=lambda x: x["discontinued"] - x["established"])
    .query("age <= 10")
    .groupby("established")
    .size()
    .sort_index()
    .plot()
)

# %%
# distribution of establishment of current post offices
(
    post.query(
        "established.notna() and discontinued.isna() and established >= 1600"
    )["established"]
    .value_counts()
    .sort_index()
    .plot()
)

# %%
# distribution of ages of post offices
(
    post.query("established.notnull()")
    .assign(discontinued=post["discontinued"].fillna(2021))
    .assign(age=lambda x: x["discontinued"] - x["established"])
    .query("age >= 0 and age < 250")
    .pipe(px.histogram, x="age", facet_col="continuous")
)

# %%
(
    post.query("established.notna()")
    .groupby(["state"])
    .agg({"established": "count", "discontinued": "count"})
    .drop(["MI/OH", "VAy"])
    .assign(total=lambda x: x["established"] - x["discontinued"])
    .reset_index()
    .pipe(
        px.choropleth,
        locations="state",
        locationmode="USA-states",
        color="total",
        scope="usa",
        labels={"total": "Total"},
    )
    # px.choropleth), locations=adh_state[‘STATEAB’], locationmode=”USA-states”,
    #  color=’PER_ADH’,color_continuous_scale=”inferno”,
    #  range_color=(0, 100),scope=”usa”,labels={‘PER_ADH’:’%Adherents’},hover_name=’State Name’,
    #  hover_data={‘STATEAB’:False,’State Name’:False,’ADHERENT’:False,’TOTPOP’:False,’PER_ADH’:True})
)

# %%
fig = (
    post.query("established.notna()")
    .assign(decade=post["established"] // 10)
    .groupby(["state", "decade"])
    .agg({"established": "count"})
    .drop(["MI/OH", "VAy"])
    .reset_index()
    .sort_values("decade")
    .query("decade >= 150 and decade < 201")
    .pipe(
        px.choropleth,
        locations="state",
        locationmode="USA-states",
        color="established",
        scope="usa",
        labels={"total": "Total"},
        range_color=(0, 1500),
        animation_frame="decade",
    )
    .show(renderer="notebook")
)
# %%
