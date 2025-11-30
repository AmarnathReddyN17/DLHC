import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Load the results you just generated
df = pd.read_csv('advanced_results.csv')
fpr, tpr, _ = roc_curve(df['True'], df['Pred'])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=3, label=f'Advanced Fusion (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Final Advanced Fusion Model (Vitals + Labs + ClinicalBERT)')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig('advanced_roc.png')
print("✅ Saved advanced_roc.png")
