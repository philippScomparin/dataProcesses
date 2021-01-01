import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from pandas_profiling import ProfileReport



df = pd.read_csv("data.csv")

print("-------------------------------------------------")
print("To get an overview of the data:")
print(df.head())
print("Number of rows:")
print(df.count())

print("-------------------------------------------------")
print("Column Types:")
print(df.dtypes)

print("-------------------------------------------------")
print("Null values:")
print(df.isnull().sum(axis=0))


print("-------------------------------------------------")
print("Correlation between predictor variables and target variable:")
corr = df.corr()
print(corr.sort_values(by=["relevant"], ascending=False).filter(items=["relevant"]))

print("-------------------------------------------------")
plt.figure(figsize(12,10))
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()
cor_target = abs(cor["relevant"])
relevant_features = cor_target[cor_target>0.2]
relevant_features


profile = ProfileReport(df, title='Pandas Profiling Report')
profile