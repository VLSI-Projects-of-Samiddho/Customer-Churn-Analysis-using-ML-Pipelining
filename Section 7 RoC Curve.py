# 7. RoC Curve analysis
plt.figure(figsize=(9, 7))
 
for name, v in results.items():
    fpr, tpr, _ = roc_curve(y_test, v["y_prob"])
    plt.plot(fpr, tpr, label=f"{name} (AUC={v['ROC-AUC']:.3f})")
 
plt.plot([0, 1], [0, 1], "--", color="gray", label="Random baseline")
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate",  fontsize=12)
plt.title("ROC Curves — All Models", fontsize=14)
plt.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig("/kaggle/working/roc_curves.png", dpi=150)
plt.show()
 