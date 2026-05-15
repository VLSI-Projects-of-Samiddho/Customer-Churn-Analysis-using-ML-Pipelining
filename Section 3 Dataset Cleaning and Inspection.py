# 3. Cleaning of Data & Encoding
df.dropna(subset=["Churn"], inplace=True)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
 
if df["Churn"].dtype == object:
    df["Churn"] = df["Churn"].str.strip().map({"Yes": 1, "No": 0})
df["Churn"] = df["Churn"].astype(int)
 
if df["Gender"].dtype == object:
    df["Gender"] = (
        df["Gender"].str.strip().str.lower().map({"male": 1, "female": 0})
    )
 
print("Shape after cleaning:", df.shape)
print("\nChurn distribution:\n", df["Churn"].value_counts())