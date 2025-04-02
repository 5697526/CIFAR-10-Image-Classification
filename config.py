DEFAULT_CONFIG = {
    'hidden_size': 1024,
    'activation': 'leaky_relu',
    'num_epochs': 100,
    'batch_size': 256,
    'learning_rate': 0.01,
    'reg_lambda': 0.01,
    'learning_rate_decay': 0.95,
    'val_ratio': 0.1,
    'save_dir': 'results'
}

CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]
