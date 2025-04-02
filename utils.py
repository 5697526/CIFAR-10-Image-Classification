import numpy as np
import matplotlib.pyplot as plt


def visualize_weights(weights, save_path=None):
    w = weights['w1']
    w = (w - w.min()) / (w.max() - w.min())

    plt.figure(figsize=(10, 10))
    for i in range(100):
        plt.subplot(10, 10, i + 1)
        plt.imshow(w[:, i].reshape(32, 32, 3))
        plt.axis('off')

    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes, save_path=None):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    if save_path:
        plt.savefig(save_path)
    plt.show()
