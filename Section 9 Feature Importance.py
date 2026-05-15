# 9. Feature Importance 
if "Random Forest" in results:
    rf_pipe  = models["Random Forest"]
    prep_fit = rf_pipe.named_steps["prep"]
    rf_model = rf_pipe.named_steps["model"]
 
    feature_names = prep_fit.get_feature_names_out()
 
    feat_df = (
        pd.DataFrame({
            "Feature":    feature_names,
            "Importance": rf_model.feature_importances_,
        })
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
 
    print("\n── Top 15 Features (Random Forest) ──")
    print(feat_df.head(15).to_string(index=False))
 
    plt.figure(figsize=(9, 6))
    sns.barplot(
        data=feat_df.head(15),
        x="Importance",
        y="Feature",
        hue="Feature",
        palette="viridis",
        legend=False,
    )
    plt.title("Top 15 Feature Importances — Random Forest", fontsize=13)
    plt.tight_layout()
    plt.savefig("/kaggle/working/feature_importance.png", dpi=150)
    plt.show()
 