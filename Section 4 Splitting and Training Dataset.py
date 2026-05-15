# 4. Splitting and Training of Dataset 
X = df.drop("Churn", axis=1)
y = df["Churn"]
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
 
print(f"Train : {X_train.shape}  |  {y_train.value_counts().to_dict()}")
print(f"Test  : {X_test.shape}   |  {y_test.value_counts().to_dict()}")