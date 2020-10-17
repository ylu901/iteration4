import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # this is used for the plot the graph
import seaborn as sns  # used for plot interactive graph.
import scipy.stats as ss

df = pd.read_csv('heart.csv')
pd.set_option('display.width',400)
pd.set_option('display.max_columns',14)
print(df)
print(df.info())
print(df.describe())

ax = sns.countplot(x="target", data=df, palette="bwr")
for p in ax.patches:
    ax.annotate(f'\n{p.get_height()}', (p.get_x() + 0.4, p.get_height()), ha='center', va='top', color='black', size=18)
ax = sns.countplot(x="fbs", data=df, palette="bwr")
plt.show()
ax = sns.countplot(x="fbs", data=df, palette="bwr")
for p in ax.patches:
    ax.annotate(f'\n{p.get_height()}', (p.get_x() + 0.4, p.get_height()), ha='center', va='top', color='black', size=18)
plt.show()
ax = sns.countplot(x="exang", data=df, palette="bwr")
for p in ax.patches:
    ax.annotate(f'\n{p.get_height()}', (p.get_x() + 0.4, p.get_height()), ha='center', va='top', color='black', size=18)
plt.show()
ax = sns.countplot(x="sex", data=df, palette="bwr")
for p in ax.patches:
    ax.annotate(f'\n{p.get_height()}', (p.get_x() + 0.4, p.get_height()), ha='center', va='top', color='black', size=18)
plt.show()

sums = df.groupby(df["cp"])["target"].sum()
plt.axis('equal')
plt.pie(sums, labels=sums.index, autopct='%1.1f%%')
plt.show()
sums = df.groupby(df["restecg"])["target"].sum()
plt.axis('equal')
plt.pie(sums, labels=sums.index, autopct='%1.1f%%')
plt.show()
sums = df.groupby(df["slope"])["target"].sum()
plt.axis('equal')
plt.pie(sums, labels=sums.index, autopct='%1.1f%%')
plt.show()
sums = df.groupby(df["thal"])["target"].sum()
plt.axis('equal')
plt.pie(sums, labels=sums.index, autopct='%1.1f%%')
plt.show()

print("age")
print("Count:"+str(len(df.age)))
print("Mean:"+str(np.nanmean(df.age)))
print("Min:"+str(np.nanmin(df.age)))
print("Max:"+str(np.nanmax(df.age)))
print("Range:"+str(np.nanmax(df.age)-np.nanmin(df.age)))
print("Variance:"+str(np.nanvar(df.age)))
print("Standard Deviation:"+str(np.nanstd(df.age)))
print("Standard Error of Mean:"+str(ss.sem(df.age)))

print("trestbps")
print("Count:"+str(len(df.trestbps)))
print("Mean:"+str(np.nanmean(df.trestbps)))
print("Min:"+str(np.nanmin(df.trestbps)))
print("Max:"+str(np.nanmax(df.trestbps)))
print("Range:"+str(np.nanmax(df.trestbps)-np.nanmin(df.trestbps)))
print("Variance:"+str(np.nanvar(df.trestbps)))
print("Standard Deviation:"+str(np.nanstd(df.trestbps)))
print("Standard Error of Mean:"+str(ss.sem(df.trestbps)))

print("thalach")
print("Count:"+str(len(df.thalach)))
print("Mean:"+str(np.nanmean(df.thalach)))
print("Min:"+str(np.nanmin(df.thalach)))
print("Max:"+str(np.nanmax(df.thalach)))
print("Range:"+str(np.nanmax(df.thalach)-np.nanmin(df.thalach)))
print("Variance:"+str(np.nanvar(df.thalach)))
print("Standard Deviation:"+str(np.nanstd(df.thalach)))
print("Standard Error of Mean:"+str(ss.sem(df.thalach)))

print("oldpeak")
print("Count:"+str(len(df.oldpeak)))
print("Mean:"+str(np.nanmean(df.oldpeak)))
print("Min:"+str(np.nanmin(df.oldpeak)))
print("Max:"+str(np.nanmax(df.oldpeak)))
print("Range:"+str(np.nanmax(df.oldpeak)-np.nanmin(df.oldpeak)))
print("Variance:"+str(np.nanvar(df.oldpeak)))
print("Standard Deviation:"+str(np.nanstd(df.oldpeak)))
print("Standard Error of Mean:"+str(ss.sem(df.oldpeak)))

print("chol")
print("Count:"+str(len(df.chol)))
print("Mean:"+str(np.nanmean(df.chol)))
print("Min:"+str(np.nanmin(df.chol)))
print("Max:"+str(np.nanmax(df.chol)))
print("Range:"+str(np.nanmax(df.chol)-np.nanmin(df.chol)))
print("Variance:"+str(np.nanvar(df.chol)))
print("Standard Deviation:"+str(np.nanstd(df.chol)))
print("Standard Error of Mean:"+str(ss.sem(df.chol)))

print("ca")
print("Count:"+str(len(df.ca)))
print("Mean:"+str(np.nanmean(df.ca)))
print("Min:"+str(np.nanmin(df.ca)))
print("Max:"+str(np.nanmax(df.ca)))
print("Range:"+str(np.nanmax(df.ca)-np.nanmin(df.ca)))
print("Variance:"+str(np.nanvar(df.ca)))
print("Standard Deviation:"+str(np.nanstd(df.ca)))
print("Standard Error of Mean:"+str(ss.sem(df.ca)))

plt.boxplot(df.trestbps)
plt.show()
plt.boxplot(df.thalach)
plt.show()
plt.boxplot(df.oldpeak)
plt.show()
plt.boxplot(df.chol)
plt.show()

# # calculate the correlation matrix
corr = df.corr()

# plot the heatmap
fig = plt.figure(figsize=(10,9))
sns.heatmap(corr,
        xticklabels=corr.columns,
        yticklabels=corr.columns,
            annot=True,
            linewidths=.85)
plt.show()