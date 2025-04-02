import numpy as np
from data_loader import load_data, preprocess_data
from model import ThreeLayerNN
import config


def evaluate(model_path, test_data, test_labels):
    input_size = test_data.shape[1]
    output_size = len(np.unique(test_labels))
    hidden_size = config.DEFAULT_CONFIG['hidden_size']

    model = ThreeLayerNN(input_size, hidden_size, output_size)
    model_parameter = np.load(model_path, allow_pickle=True).item()

    model.w1 = model_parameter['w1']
    model.b1 = model_parameter['b1']
    model.w2 = model_parameter['w2']
    model.b2 = model_parameter['b2']
    model.w3 = model_parameter['w3']
    model.b3 = model_parameter['b3']

    test_accuracy = model.accuracy(test_data, test_labels)
    print(f"Test Accuracy: {test_accuracy:.4f}")


if __name__ == '__main__':
    train_data, train_labels, test_data, test_labels = load_data()
    _, test_data = preprocess_data(train_data, test_data)

    evaluate('results/best_model.npy', test_data, test_labels)
