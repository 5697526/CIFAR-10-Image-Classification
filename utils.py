import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_data, preprocess_data
from model import ThreeLayerNN
import config


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


if __name__ == '__main__':
    train_data, train_labels, test_data, test_labels = load_data()
    train_data, test_data = preprocess_data(train_data, test_data)
    input_size = test_data.shape[1]
    output_size = len(np.unique(test_labels))
    hidden_size = config.DEFAULT_CONFIG['hidden_size']
    model = ThreeLayerNN(input_size, hidden_size, output_size,
                         config.DEFAULT_CONFIG['activation'])
    model_parameter = np.load('results/best_model.npy',
                              allow_pickle=True).item()
    model.w1 = model_parameter['w1']
    model.b1 = model_parameter['b1']
    model.w2 = model_parameter['w2']
    model.b2 = model_parameter['b2']
    model.w3 = model_parameter['w3']
    model.b3 = model_parameter['b3']

    y_pred = model.predict(test_data)

    visualize_weights(
        model_parameter, save_path='results/weights_visualization.png')
    plot_confusion_matrix(test_labels, y_pred, config.CLASS_NAMES,
                          save_path='results/confusion_matrix.png')
