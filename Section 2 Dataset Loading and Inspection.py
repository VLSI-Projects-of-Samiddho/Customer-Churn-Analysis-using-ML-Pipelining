# 2. Loading of Dataset & Inspection 
df = pd.read_csv(
    "/kaggle/input/customer-churn-dataset/"
    "customer_churn_dataset-training-master.csv"
)
 
print("Shape      :", df.shape)
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isna().sum())
print("\nTarget distribution:\n", df["Churn"].value_counts(dropna=False))
df.head()