# %%
import pandas as pd
import numpy as np
import sklearn as sk
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('ggplot')

# %%
train = pd.read_csv('train.csv')
target = train.Survived.astype(np.float32)
# train = train.drop('Survived', axis=1)
test = pd.read_csv('test.csv')
train.head()

# %%
N = train.shape[0]
# df = pd.concat([train, test], axis=0)
df = train

df['Title'] = df['Name'].map(lambda x: x.split(
    ',')[1].split('.')[0].split()[-1])

df['Exotic'] = ~df['Title'].isin(['Mr', 'Miss', 'Mrs', 'Master'])
df['Title'] = np.where(df['Exotic'], 'Exotic', df['Title'])

# %%
train['Cabin'] = (train.groupby('Ticket')['Cabin']
                  .apply(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
                  )

# %%
df = pd.get_dummies(
    df, columns=['Embarked', 'Pclass', 'Sex', 'Title'])
train = df[:N]
test = df[N:]
train.dtypes

# %%
sns.heatmap(pd.DataFrame(target).join(train.drop('Survived', axis=1)).drop(
    'PassengerId', axis=1).corr(), cmap='jet')
# %%
print(train.isnull().sum())
print(test.isnull().sum())

# %% [markdown]
# Try to detect age.  We compare age to sex, passanger class, embarked, and fare.

# %%
train.pipe((sns.FacetGrid, 'data'), col='Sex').map(sns.histplot, 'Age')
train.groupby('Sex')['Age'].mean()

# %%
train.pipe((sns.FacetGrid, 'data'), col='Pclass').map(sns.histplot, 'Age')
train.groupby('Pclass')['Age'].mean()

# %%
train.pipe((sns.FacetGrid, 'data'), col='Embarked').map(sns.histplot, 'Age')
train.groupby('Embarked')['Age'].mean()

# %%
train.pipe((sns.FacetGrid, 'data'), sharex=False, sharey=False,
           col='Pclass').map(sns.histplot, 'Fare')
train.groupby('Pclass')['Fare'].mean()

# %%
sns.scatterplot(data=train, x='Fare', y='Age')

# %%
train[train['Pclass'].isin([2, 3])]['Fare'].max()
# %%
train[train['Age'].isnull()]['Fare'].hist()

# %%
train.groupby('Pclass')['Cabin'].apply(lambda x: x.isnull().sum())

# %%


def process(df):
    if 'Survived' in df.columns:
        df = df.drop('Survived', axis=1)

    return (df
            .assign(NoAge=df.Age.isnull())
            .fillna(value={'Age': df.Age.mean(), 'Embarked': 'S', 'Fare': df.Fare.mean()})
            .assign(Cabin=df.Cabin.isnull())
            .drop(['PassengerId', 'Ticket', 'Name'], axis=1)
            )


train = process(train).astype(np.float32)
train.dtypes

# %%
import tensorflow as tf


def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])


# %%
import datetime
%load_ext tensorboard

model = build_model()
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-2),
              loss='binary_crossentropy',
              metrics=['accuracy'])

path = 'logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=path, histogram_freq=1)

# %%
history = model.fit(train, target, epochs=100,
                    validation_split=0.1, callbacks=[tensorboard_callback])

# %%
pd.DataFrame(history.history).plot(figsize=(8, 6))

# %%
preds = model.predict(test.to_numpy().astype(np.float64)).round()

# %%
x = pd.read_csv('test.csv')
sub = pd.DataFrame(
    {'PassengerId': x['PassengerId'], 'Survived': preds.astype(int).T[0]})
sub.to_csv('submission.csv', index=False)
sub.head()
