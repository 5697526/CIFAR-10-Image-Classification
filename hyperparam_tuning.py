import numpy as np
import itertools
from train import training
from data_loader import load_data, preprocess_data, validation


def hyperparameter():
    train_data, train_labels, test_data, test_labels = load_data()
    train_data, test_data = preprocess_data(train_data, test_data)
    train_data, train_labels, val_data, val_labels = validation(
        train_data, train_labels)

    hidden_sizes = [256, 512, 1024]
    learning_rates = [0.001, 0.01]
    reg_lambdas = [0.001, 0.01, 0.1]
    activations = ['relu', 'tanh', 'leaky_relu']

    results = []

    for hidden_size, lr, reg, activation in itertools.product(
            hidden_sizes, learning_rates, reg_lambdas, activations):

        print(
            f"\nTraining with hidden_size={hidden_size}, lr={lr}, reg={reg}, activation={activation}")

        config = {
            'hidden_size': hidden_size,
            'activation': activation,
            'num_epochs': 20,
            'batch_size': 256,
            'learning_rate': lr,
            'reg_lambda': reg,
            'learning_rate_decay': 0.95
        }

        _, history = training(
            train_data, train_labels, val_data, val_labels, config)

        best_val_acc = max(history['accuracy'])
        results.append({
            'hidden_size': hidden_size,
            'learning_rate': lr,
            'reg_lambda': reg,
            'activation': activation,
            'val_acc': best_val_acc
        })

        print(f"Best validation accuracy: {best_val_acc:.4f}")

    best_result = max(results, key=lambda x: x['val_acc'])
    print("\nBest Hyperparameters:")
    print(f"Hidden Size: {best_result['hidden_size']}")
    print(f"Learning Rate: {best_result['learning_rate']}")
    print(f"Regularization: {best_result['reg_lambda']}")
    print(f"Activation: {best_result['activation']}")
    print(f"Validation Accuracy: {best_result['val_acc']:.4f}")

    return results, best_result


if __name__ == '__main__':
    hyperparameter()
