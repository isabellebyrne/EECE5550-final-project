import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
from data import get_datasets
import numpy as np


def plot_precision_recall(models, model_names, x_test, y_test):
    plt.figure(figsize=(10, 7))
    
    for model, model_name in zip(models, model_names):
        precision, recall, _ = precision_recall_curve(y_test, model.predict(x_test))
        auc_score = auc(recall, precision)
        plt.plot(recall, precision, label=f'{model_name} (AUC = {auc_score:.2f})')
        
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision Recall Curve')
    plt.legend(loc='best')
    plt.grid(alpha=0.2)
    plt.show()