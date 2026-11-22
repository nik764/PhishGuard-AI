
import pandas as pd
from urllib.parse import urlsplit

df = pd.read_csv("dataset/combined_urls.csv")

df["HasPath"] = df["URL"].apply(
    lambda url: bool(urlsplit(str(url)).path.strip("/"))
)

df["HasQuery"] = df["URL"].apply(
    lambda url: bool(urlsplit(str(url)).query)
)

print("\n========== COMBINED DATASET PATH DISTRIBUTION ==========")
print("0 = Phishing, 1 = Legitimate")
print(pd.crosstab(df["HasPath"], df["label"]))

print("\n========== COMBINED DATASET QUERY DISTRIBUTION ==========")
print(pd.crosstab(df["HasQuery"], df["label"]))

print("\n========== LEGITIMATE URL COUNTS ==========")
print("Legitimate URLs with paths:",
      len(df[(df["label"] == 1) & (df["HasPath"])]))

print("Legitimate URLs with queries:",
      len(df[(df["label"] == 1) & (df["HasQuery"])]))

print("\n========== DETAILED URL CATEGORY DISTRIBUTION ==========")

df["URL_Category"] = "Homepage"

df.loc[df["HasPath"], "URL_Category"] = "Path"
df.loc[df["HasQuery"], "URL_Category"] = "Query"

df.loc[
    df["HasPath"] & df["HasQuery"],
    "URL_Category"
] = "Path + Query"

print(
    pd.crosstab(
        df["URL_Category"],
        df["label"],
        margins=True
    )
)
