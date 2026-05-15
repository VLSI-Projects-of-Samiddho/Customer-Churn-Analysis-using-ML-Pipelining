# 8. Confusion Matrices of individual Algorithms
n     = len(results)
ncols = 4
nrows = -(-n // ncols)
 
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
axes = axes.flatten()
 
for ax, (name, v) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, v["y_pred"])
    ConfusionMatrixDisplay(
        cm, display_labels=["No Churn", "Churn"]
    ).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name, fontsize=9)
 
for ax in axes[n:]:
    ax.set_visible(False)
 
plt.suptitle("Confusion Matrices — All Models", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("/kaggle/working/confusion_matrices.png", dpi=150)
plt.show()
 