import numpy as np
import itertools
import os
import json
from train import training
from data_loader import load_data, preprocess_data, validation


def save_results(results, best_result, save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)

    json_path = os.path.join(save_dir, 'hyperparam_results_3.json')
    with open(json_path, 'w') as f:
        json.dump({
            'all_results': results,
            'best_result': best_result
        }, f, indent=2)

    best_path = os.path.join(save_dir, 'best_result.json')
    with open(best_path, 'w') as f:
        json.dump(best_result, f, indent=2)

    print(f"\nResults saved to:")
    print(f"- JSON: {json_path}")
    print(f"- Best result: {best_path}")


def hyperparameter():
    hidden_sizes = [512]
    learning_rates = [0.018, 0.019, 0.017]
    reg_lambdas = [0.011]
    activations = ['leaky_relu']
    learning_rate_decays = [0.98]

    num_epochs = 20
    batch_size = 256

    try:
        train_data, train_labels, test_data, test_labels = load_data()
        train_data, test_data = preprocess_data(train_data, test_data)
        train_data, train_labels, val_data, val_labels = validation(
            train_data, train_labels)
    except Exception as e:
        print(f"Error loading data: {e}")
        return [], {}

    results = []
    total_combinations = len(hidden_sizes) * len(learning_rates) * \
        len(reg_lambdas) * len(activations) * len(learning_rate_decays)
    current = 0

    print(
        f"\nStarting hyperparameter search ({total_combinations} combinations)...")

    for hidden_size, lr, reg, activation, decay in itertools.product(
            hidden_sizes, learning_rates, reg_lambdas, activations, learning_rate_decays):

        current += 1
        print(f"\n[{current}/{total_combinations}] Testing: "
              f"hidden_size={hidden_size}, lr={lr:.3f}, reg={reg:.3f}, activation={activation}, decay={decay}")

        config = {
            'hidden_size': hidden_size,
            'activation': activation,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': lr,
            'reg_lambda': reg,
            'learning_rate_decay': decay
        }

        try:
            _, history = training(
                train_data, train_labels, val_data, val_labels, config)
        except Exception as e:
            print(f"Error training model with hyperparameters {config}: {e}")
            continue

        best_val_acc = max(history['accuracy'])
        results.append({
            'hidden_size': hidden_size,
            'learning_rate': lr,
            'reg_lambda': reg,
            'activation': activation,
            'learning_rate_decay': decay,
            'val_acc': best_val_acc,
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1]
        })

        print(f"Best validation accuracy: {best_val_acc:.4f}")

    if not results:
        print("No valid results obtained.")
        return [], {}

    best_result = max(results, key=lambda x: x['val_acc'])

    print("\n=== Search Summary ===")
    print(f"Total combinations tested: {len(results)}")
    print("\nTop 3 Performers:")
    for i, r in enumerate(sorted(results, key=lambda x: -x['val_acc'])[:3]):
        print(f"{i + 1}. val_acc={r['val_acc']:.4f} (hidden_size={r['hidden_size']}, "
              f"lr={r['learning_rate']:.3f}, reg={r['reg_lambda']:.3f}, "
              f"activation={r['activation']}, decay={r['learning_rate_decay']})")

    print("\nBest Hyperparameters:")
    print(f"Hidden Size: {best_result['hidden_size']}")
    print(f"Learning Rate: {best_result['learning_rate']:.3f}")
    print(f"Regularization: {best_result['reg_lambda']:.3f}")
    print(f"Activation: {best_result['activation']}")
    print(f"Learning Rate Decay: {best_result['learning_rate_decay']}")
    print(f"Validation Accuracy: {best_result['val_acc']:.4f}")

    save_results(results, best_result)

    return results, best_result


if __name__ == '__main__':
    hyperparameter()
