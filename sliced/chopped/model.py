# %%
import pandas as pd
import seaborn as sns
import numpy as np
from plotnine import *
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from tidytuesday import TidyTuesday
from myprelude import seed_everything

# %%
tt = TidyTuesday("2020-08-25")
df = tt.data["chopped"].query("episode_rating.notna()")

# %%
from sklearn.model_selection import train_test_split

seed_everything(42069)
train, test = train_test_split(df, test_size=0.2)
# %%
df["season"].hist()
# %%
train["episode_rating"].hist()
# %%
train["episode_rating"].std()
# %%
sns.boxplot(data=train, x="season", y="episode_rating", orient="v")

# %%
(
    train["entree"].str.split(",")
    + train["dessert"].str.split(",")
    + train["appetizer"].str.split(",")
).explode().value_counts().value_counts()

# %%
sns.countplot(
    y=fct_lump(
        pd.Series(
            train[["judge1", "judge2", "judge3"]].values.tolist()
        ).explode()
    ),
    orient="v",
)

# %%
pd.Series(
    train[["judge1", "judge2", "judge3"]].values.tolist()
).explode().value_counts().where(lambda x: x > 1).dropna()
# %%

from myprelude.trans import fct_lump

judges = (
    train[["judge1", "judge2", "judge3"]]
    .reset_index()
    .melt("index")
    .assign(value=lambda x: fct_lump(x["value"]), vals=True)
    .query('value != "Other"')
    .pivot("index", "value", "vals")
    .fillna(False)
    .astype("int32")
)

# %%
from nltk.corpus import stopwords

stop = stopwords.words("english")
(
    train.assign(
        notes=train["episode_notes"]
        .str.lower()
        .str.replace("[,\.]", "")
        .str.split(" ")
    )
    .explode("notes")
    .query("notes not in @stop")["notes"]
    .value_counts()
)

# %%
pd.concat([train[["episode_rating"]], judges], axis=1).corr().iloc[
    1:, 0
].plot.bar()

# %%
(
    ggplot(train, aes("series_episode", "episode_rating"))
    + geom_point(aes(color="season"))
    + geom_smooth(method="loess")
)
# %%
(
    ggplot(train, aes("season_episode", "episode_rating"))
    + geom_point()
    + geom_smooth(method="loess")
)

# %%
from patsy import cr
from sklearn.linear_model import LinearRegression

mse = []

for d in range(3, 8):
    model = LinearRegression().fit(
        cr(train["series_episode"], df=d), train["episode_rating"]
    )
    mse.append(
        np.sqrt(
            np.mean(
                (
                    train["episode_rating"]
                    - model.predict(cr(train["series_episode"], df=d))
                )
                ** 2
            )
        )
    )


# (
#     ggplot()
#     + geom_point(aes(train["series_episode"], train["episode_rating"]))
#     + geom_line(
#         aes(
#             train["series_episode"],
#             model.predict(cr(train["series_episode"], df=6)),
#         ),
#         color="red",
#     )
# )


# %%
train = train.join(cr(train["series_episode"], df=6))
test = test.join(cr(test["series_episode"], df=6))

# %%
(
    train.assign(
        judges=train[["judge1", "judge2", "judge3"]].apply(
            lambda x: list(x), axis=1
        )
    )
)


# %%
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression


class JudgeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n=10):
        self.judges = []
        self.n = 10

    def melt(self, X):
        return (
            X[["judge1", "judge2", "judge3"]]
            .reset_index()
            .melt("index")[["index", "value"]]
        )

    def pivot(self, X):
        return (
            X.query("value in @self.judges")
            .groupby("index")
            .apply(lambda x: list(x["value"]))
        )

    def fit(self, X):
        judge_list = self.melt(X)

        self.judges = judge_list["value"].value_counts()[:10].index.to_list()
        self.enc = TfidfVectorizer(analyzer=lambda x: x, use_idf=False).fit(
            self.pivot(judge_list)
        )

        return self

    def transform(self, X):
        return self.enc.transform(self.pivot(self.melt(X)))


print(JudgeTransformer().fit_transform(train).todense().shape)
print(len(train))

# col = ColumnTransformer(['judges', None, ['judge1', 'judge2', 'judge3']])
# %%
from sklearn.feature_extraction.text import CountVectorizer
