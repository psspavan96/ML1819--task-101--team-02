import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("./dataset.orig/google-play-store-apps/googleplaystore.csv")
print("Number of data points:", df.shape[0])

# df.head()

# df.info()

df = df[df["Rating"]<=5]

df.Category.unique()

CategoryString = df["Category"]
categoryVal = df["Category"].unique()
categoryValCount = len(categoryVal)
category_dict = {}

for i in range(0,categoryValCount):
    category_dict[categoryVal[i]] = i
df["Category"] = df["Category"].map(category_dict).astype(int)

df["Genres"].unique()

genresString = df["Genres"]
genresVal = df["Genres"].unique()
genresValCount = len(genresVal)
genres_dict = {}
for i in range(0,genresValCount):
    genres_dict[genresVal[i]] = i
df["Genres"] = df["Genres"].map(genres_dict).astype(int)

df['Content Rating'].unique()

df['Content Rating'] = df['Content Rating'].map({'Everyone':0,'Teen':1,'Everyone 10+':2,'Mature 17+':3,'Adults only 18+':4}).astype(float)

df['Reviews'] = [ float(i.split('M')[0]) if 'M'in i  else float(i) for i in df['Reviews']]

df["Size"] = [ float(i.split('M')[0]) if 'M' in i else float(0) for i in df["Size"]  ]

df['Price'] = [ float(i.split('$')[1]) if '$' in i else float(0) for i in df['Price'] ]

df.Installs.unique()

df["Installs"] = [ float(i.replace('+','').replace(',', '')) if '+' in i or ',' in i else float(0) for i in df["Installs"] ]

df.drop(["Last Updated","Current Ver","Android Ver","App","Type"],axis=1,inplace=True)

df["Rating"] = df.groupby("Category")["Rating"].transform(lambda x: x.fillna(x.mean()))
df["Content Rating"] = df[["Content Rating"]].fillna(method="ffill")
df["Rated 4.4 or more"] = [ 1 if (i >= 4.4) else -1 for i in df['Rating'] ]

# df.info()

# print (df.shape)

# print (df.head())

# print (df.describe())

# X = df.drop(["Rating"],axis=1)
# y = df.Rating
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)

df.to_csv("./dataset.preprocessed/googleplay/cleaned.csv")
# X_train.to_csv("./dataset.preprocessed/X_train.csv")
# y_train.to_csv("./dataset.preprocessed/y_train.csv")
# X_test.to_csv("./dataset.preprocessed/X_test.csv")
# y_test.to_csv("./dataset.preprocessed/y_test.csv")

# np.savetxt("./dataset.preprocessed/googleplay/cleaned.csv", df, delimiter=",")
# np.savetxt("./dataset.preprocessed/X_train.csv", X_train, delimiter=",")
# np.savetxt("./dataset.preprocessed/y_train.csv", y_train, delimiter=",")
# np.savetxt("./dataset.preprocessed/X_test.csv", X_test, delimiter=",")
# np.savetxt("./dataset.preprocessed/y_test.csv", y_test, delimiter=",")