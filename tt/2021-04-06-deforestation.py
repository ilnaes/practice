# %%
from tidytuesday import TidyTuesday
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
from myprelude import *
from myprelude.trans import fct_lump, fct_reorder, FctLumper


# %%
tt = TidyTuesday("2021-04-06")
# %%
forest_area = tt.data["forest_area"]
# %%
forest_area_code = forest_area.query("code.notnull()", engine="python")
forest_area_code = forest_area_code[
    forest_area_code["code"].apply(len) == 3
].drop(["code"], axis=1)
# %%
(
    forest_area_code.query("year == [1992,2020]")
    .pivot(["entity"], "year", "forest_area")
    .dropna()
    .assign(delta=lambda x: (x[2020] - x[1992]) / x[1992])
    .dropna()
    .sort_values("delta")
    .pipe((sns.kdeplot, "data"), x="delta")
)
# %%
(
    forest_area_code.query("year == [1992,2020]")
    .pivot(["entity"], "year", "forest_area")
    .dropna()
    .assign(delta=lambda x: (x[2020] - x[1992]) / x[1992])
    .dropna()
    .sort_values("delta")
    .pipe(px.histogram, x="delta")
)
# %%
cats = (
    forest_area_code.groupby("entity")
    .agg({"forest_area": "mean"})
    .sort_values("forest_area", ascending=False)["forest_area"][:9]
    .index.tolist()
)
# %%
order = (
    forest_area_code.query("year >= 1992")
    .assign(entity=lambda x: x["entity"].where(x["entity"].isin(cats), "Other"))
    .groupby(["entity", "year"])
    .agg({"forest_area": "sum"})
    .groupby("entity")
    .agg({"forest_area": "mean"})
    .sort_values("forest_area", ascending=False)["forest_area"]
    .index.tolist()
)

(
    forest_area_code.query("year >= 1992")
    .assign(entity=lambda x: x["entity"].where(x["entity"].isin(cats), "Other"))
    .groupby(["entity", "year"])
    .agg({"forest_area": "sum"})
    # .loc[order, :]
    .unstack()
    .T.loc["forest_area"][order]
    .plot.area()
    .legend(loc="center right", bbox_to_anchor=(1.3, 0.5), ncol=1)
    # .pipe((sns.lineplot, "data"), x="year", y="forest_area", hue="entity")
    # .legend(loc="center right", bbox_to_anchor=(1.6, 0.5), ncol=1)
)

# %%
(
    forest_area_code.query("year >= 1992")
    .pivot("entity", "year", "forest_area")
    .dropna()
    .apply(lambda x: x / x[1992], axis=1)
    .dropna()
    .sort_values(2020)
    # .assign(c=forest_area_code["entity"].map(lambda x: x if x in cats else "Other"))
)
# %%
forest = tt.data["forest"]
# %%
forest_code = forest.query("code.notnull()", engine="python")
forest_code = forest_code[forest_code["code"].map(len) == 3].drop(
    ["code"], axis=1
)
# %%
(
    forest_code.groupby("entity")
    .agg({"net_forest_conversion": "sum"})
    .sort_values("net_forest_conversion", key=np.abs, ascending=False)[:10]
    .sort_values("net_forest_conversion", ascending=False)
    .reset_index()
    .assign(pos=lambda x: x["net_forest_conversion"] > 0)
    .pipe(
        (sns.barplot, "data"), y="entity", x="net_forest_conversion", hue="pos"
    )
)

# %%
(
    forest_code.groupby("year")
    .apply(
        lambda x: x.sort_values(
            "net_forest_conversion", key=np.abs, ascending=False
        )[:10].sort_values("net_forest_conversion", ascending=False)
    )
    .reset_index(drop=True)
    .assign(pos=lambda x: np.sign(x["net_forest_conversion"]))
    .pipe((sns.FacetGrid, "data"), col="year", height=5, col_wrap=2)
    .map(sns.barplot, "net_forest_conversion", "entity", "pos")
)

# %%
brazil = tt.data["brazil_loss"].drop(["code", "entity"], axis=1)
# %%
brazil_loss = brazil.melt(
    id_vars="year", var_name="cause", value_name="loss"
).sort_values("loss", ascending=False)

# %%
brazil.melt("year").assign(
    variable=lambda x: fct_reorder(x["variable"], x["value"])
)["variable"]

# %%
(
    brazil.drop(["pasture"], axis=1)
    .set_index("year")
    .T.sort_values(by=2001)
    .T.plot()
)

# %%
fct = FctLumper()
(
    fct.fit_transform(brazil_loss, col="cause", w="loss").assign(
        cause=lambda x: fct_reorder(x["cause"], -x["loss"])
    )["cause"]
    # .pipe(px.area, x="year", y="loss", color="cause")
)
# %%
fct.get_params()
# %%
soybean = tt.data["soybean_use"]
# %%
soybean = soybean.query("code.notnull()", engine="python")
soybean = soybean[soybean["code"].map(len) == 3].drop(["code"], axis=1)
# %%
(soybean.fillna(method="ffill").fillna(0).groupby("year").sum().plot.area())

# %%
soybean_fill = soybean.fillna(method="ffill").fillna(0)

# %%
total = (
    soybean_fill.query("year == 2013")
    .assign(sum=lambda x: x.iloc[:, 2:5].sum(axis=1))
    .set_index("entity")["sum"]
)

# %%
(
    soybean_fill.melt(["entity", "year"])
    .query(
        'entity not in ["Brunei", "Armenia", "Colombia", "Costa Rica", "Congo", "Cote d\'Ivoire"]'
    )
    .assign(entity=lambda x: fct_lump(x["entity"], 6, x["value"]))
    .groupby(["entity", "year"])
    .sum()
    .reset_index()
    .pivot("entity", "year", "value")
    .reset_index()
    .sort_values(2013)
    .melt("entity")
    .iloc[::-1]
    .pipe((sns.lineplot, "data"), x="year", y="value", hue="entity")
)
# %%
# %%
# %%
soybean_fill.melt(["entity", "year"]).pipe(
    (sns.FacetGrid, "data"), col="variable"
).pipe()

# %%
