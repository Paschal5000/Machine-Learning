# %%
import pandas as pd
import numpy as np
from sklearn import linear_model
from word2number import w2n

# %%
df = pd.read_csv("C:/Users/pasch/Book1.csv")
df

# %%
def safe_convert(x):
    try:
        return w2n.word_to_num(str(x))
    except:
        return None # or 0, or x
df['Experience'] = df['Experience'].apply(safe_convert)

    

# %%
df

# %%
df['Experience'] = df['Experience'].fillna(0)

# %%
df

# %%
df.drop(columns=['Expereince'], inplace=True)

# %%
df

# %%
df['test_score(out of 10)'].fillna, inplace=True)(df['test_score(out of 10)'].median()

# %%
df

# %%
df['test_score(out of 10)'].median()

# %%
reg = linear_model.LinearRegression()
reg.fit(df[['Experience','test_score(out of 10)','interview_score(out of 10)']],df['salary($)'])

# %%
reg.predict([[2,9,6]])

# %%
reg.predict([[12,10,10]])

# %%



