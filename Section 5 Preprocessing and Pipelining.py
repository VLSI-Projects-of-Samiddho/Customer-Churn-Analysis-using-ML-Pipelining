# 5. Preprocessing & Pipelining
def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                make_column_selector(dtype_include=np.number),
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                make_column_selector(dtype_include=object),
            ),
        ],
        remainder="drop",
    )
 

models = {
    "Logistic Regression": Pipeline([
        ("prep", make_preprocessor()),
        ("model", LogisticRegression(max_iter=1000, random_state=42)),
    ]),
    "Naive Bayes": Pipeline([
        ("prep", make_preprocessor()),
        ("model", GaussianNB()),
    ]),
    "Decision Tree": Pipeline([
        ("prep", make_preprocessor()),
        ("model", DecisionTreeClassifier(random_state=42)),
    ]),
    "Random Forest": Pipeline([
        ("prep", make_preprocessor()),
        ("model", RandomForestClassifier(
            n_estimators=300, n_jobs=-1, random_state=42
        )),
    ]),
    "Gradient Boosting": Pipeline([
        ("prep", make_preprocessor()),
        ("model", GradientBoostingClassifier(random_state=42)),
    ]),
    "XGBoost": Pipeline([
        ("prep", make_preprocessor()),
        ("model", XGBClassifier(
            eval_metric="logloss", n_jobs=-1, random_state=42
        )),
    ]),
    "LightGBM": Pipeline([
        ("prep", make_preprocessor()),
        ("model", LGBMClassifier(n_jobs=-1, random_state=42)),
    ]),
    "CatBoost": Pipeline([
        ("prep", make_preprocessor()),
        ("model", CatBoostClassifier(verbose=0, random_state=42)),
    ]),
}
 
print(f"{len(models)} pipelines defined.")