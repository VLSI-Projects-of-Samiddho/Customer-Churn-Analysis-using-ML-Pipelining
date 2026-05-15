# 6. Training & Evaluation
 
for name, pipe in models.items():
    print(f"\nTraining: {name} ...", end=" ", flush=True)
    t0 = time.time()
 
    try:
        pipe.fit(X_train, y_train)
        elapsed = time.time() - t0
 
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
 
        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "ROC-AUC":  roc_auc_score(y_test, y_prob),
            "y_pred":   y_pred,
            "y_prob":   y_prob,
        }
        print(f"done in {elapsed:.1f}s  |  "
              f"Acc={results[name]['Accuracy']:.4f}  "
              f"AUC={results[name]['ROC-AUC']:.4f}")
        print(classification_report(y_test, y_pred))
 
    except Exception as e:
        print(f"FAILED — {e}")
        continue
 
# Summary table
summary = (
    pd.DataFrame(
        {n: {"Accuracy": v["Accuracy"], "ROC-AUC": v["ROC-AUC"]}
         for n, v in results.items()}
    )
    .T
    .sort_values("ROC-AUC", ascending=False)
)
print("\n── Model Comparison (sorted by ROC-AUC) ──")
print(summary.round(4).to_string())