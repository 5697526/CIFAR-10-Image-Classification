import numpy as np
import matplotlib.pyplot as plt
import os
from model import ThreeLayerNN
from data_loader import load_data, preprocess_data, validation


def training(train_data, train_labels, val_data, val_labels, config):
    input_size = train_data.shape[1]
    output_size = len(np.unique(train_labels))

    model = ThreeLayerNN(
        input_size, config['hidden_size'], output_size, config['activation'])

    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    reg_lambda = config['reg_lambda']
    learning_rate_decay = config['learning_rate_decay']

    history = {
        'train_loss': [],
        'val_loss': [],
        'accuracy': [],
        'best_accuracy': 0.0,
        'best_weights': None
    }

    n = train_data.shape[0]
    iterations = max(n // batch_size, 1)

    for epoch in range(num_epochs):
        if epoch % 10 == 0 and epoch > 0:
            learning_rate *= learning_rate_decay

        indices = np.random.permutation(n)
        train_data = train_data[indices]
        train_labels = train_labels[indices]

        for i in range(iterations):
            start_i = i * batch_size
            end_i = min((i+1) * batch_size, n)
            batch_data = train_data[start_i:end_i]
            batch_labels = train_labels[start_i:end_i]

            model.forward(batch_data)
            dw1, db1, dw2, db2, dw3, db3 = model.backward(
                batch_data, batch_labels, reg_lambda)

            model.w1 -= learning_rate * dw1
            model.b1 -= learning_rate * db1
            model.w2 -= learning_rate * dw2
            model.b2 -= learning_rate * db2
            model.w3 -= learning_rate * dw3
            model.b3 -= learning_rate * db3

        train_loss = model.loss_fun(train_data, train_labels, reg_lambda)
        val_loss = model.loss_fun(val_data, val_labels, reg_lambda)
        accuracy = model.accuracy(val_data, val_labels)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['accuracy'].append(accuracy)

        if accuracy > history['best_accuracy']:
            history['best_accuracy'] = accuracy
            history['best_weights'] = {
                'w1': model.w1.copy(),
                'b1': model.b1.copy(),
                'w2': model.w2.copy(),
                'b2': model.b2.copy(),
                'w3': model.w3.copy(),
                'b3': model.b3.copy()
            }
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{num_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, accuracy={accuracy:.4f}")
    return model, history


def plot_training(history, save_path=None):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Validation Accuracy')

    if save_path:
        plt.savefig(save_path)
    plt.show()


def save_model(model, path):
    np.save(path, {
        'w1': model.w1,
        'b1': model.b1,
        'w2': model.w2,
        'b2': model.b2,
        'w3': model.w3,
        'b3': model.b3,
        'activation': model.activation
    })


def load_model(path, input_size, hidden_size, output_size):
    data = np.load(path, allow_pickle=True).item()
    model = ThreeLayerNN(input_size, hidden_size,
                         output_size, data['activation'])
    model.w1 = data['w1']
    model.b1 = data['b1']
    model.w2 = data['w2']
    model.b2 = data['b2']
    model.w3 = data['w3']
    model.b3 = data['b3']
    return model


if __name__ == '__main__':
    train_data, train_labels, test_data, test_labels = load_data()
    train_data, test_data = preprocess_data(train_data, test_data)
    train_data, train_labels, val_data, val_labels = validation(
        train_data, train_labels)

    config = {
        'hidden_size': 1024,
        'activation': 'leaky_relu',
        'num_epochs': 100,
        'batch_size': 256,
        'learning_rate': 0.01,
        'reg_lambda': 0.01,
        'learning_rate_decay': 0.95
    }

    model, history = training(
        train_data, train_labels, val_data, val_labels, config)

    os.makedirs('results', exist_ok=True)
    save_model(model, 'results/best_model.npy')
    plot_training(history, 'results/training_curve.png')

    test_acc = model.accuracy(test_data, test_labels)
    print(f"Test Accuracy: {test_acc:.4f}")
