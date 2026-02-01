

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

print("TensorFlow version:", tf.__version__)

# ============================================================================
# CUSTOM LAYERS IMPLEMENTATION
# ============================================================================

class CustomDenseReluLayer(tf.keras.layers.Layer):
    """Custom Dense Layer with ReLU activation"""
    def __init__(self, units):
        super(CustomDenseReluLayer, self).__init__()
        self.units = units
    
    def build(self, input_shape):
        # Initialize weights using He initialization (good for ReLU)
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='he_normal',
            trainable=True,
            name='weights'
        )
        # Initialize biases to zero
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name='biases'
        )
    
    def call(self, inputs):
        # Linear transformation followed by ReLU
        z = tf.matmul(inputs, self.w) + self.b
        return tf.nn.relu(z)


class CustomDenseSoftmaxLayer(tf.keras.layers.Layer):
    """Custom Dense Layer with Softmax activation"""
    def __init__(self, units):
        super(CustomDenseSoftmaxLayer, self).__init__()
        self.units = units
    
    def build(self, input_shape):
        # Initialize weights using Glorot (Xavier) initialization
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True,
            name='weights'
        )
        # Initialize biases to zero
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name='biases'
        )
    
    def call(self, inputs):
        # Linear transformation followed by Softmax
        z = tf.matmul(inputs, self.w) + self.b
        return tf.nn.softmax(z)


class CustomFlattenLayer(tf.keras.layers.Layer):
    """Custom Flatten Layer"""
    def call(self, inputs):
        # Reshape to (batch_size, -1)
        batch_size = tf.shape(inputs)[0]
        return tf.reshape(inputs, [batch_size, -1])


# ============================================================================
# PART 1: MNIST with Custom Layers
# ============================================================================

print("\n" + "="*70)
print("PART 1: MNIST Classification with Custom Layers")
print("="*70)

# Load and preprocess MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train_mnist = x_train / 255.0
x_test_mnist = x_test / 255.0
y_train_mnist = to_categorical(y_train, 10)
y_test_mnist = to_categorical(y_test, 10)

print(f"\nDataset shape: {x_train_mnist.shape}")
print(f"Number of classes: {y_train_mnist.shape[1]}")

# Visualize samples
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for i in range(3):
    axes[i].imshow(x_train[i], cmap='Greys')
    axes[i].set_title(f'Digit: {y_train[i]}')
    axes[i].axis('off')
plt.tight_layout()
plt.savefig('mnist_samples.png', dpi=150, bbox_inches='tight')
print("✓ Saved sample images")

# Build model with custom layers
model_custom = Sequential([
    CustomFlattenLayer(),
    CustomDenseReluLayer(128),
    CustomDenseSoftmaxLayer(10)
])

# Compile and train
model_custom.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nTraining model...")
history_custom = model_custom.fit(
    x_train_mnist, y_train_mnist,
    epochs=5,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluate
test_loss, test_acc = model_custom.evaluate(x_test_mnist, y_test_mnist, verbose=0)
print(f'\n✓ Test accuracy: {test_acc:.4f}')


# ============================================================================
# PART 2: Boston Housing - Data Loading
# ============================================================================

print("\n" + "="*70)
print("PART 2: Housing Price Prediction")
print("="*70)

# Load housing dataset
try:
    from sklearn.datasets import load_boston
    print("\nUsing Boston Housing dataset")
    boston = load_boston()
    x_housing = boston.data
    y_housing = boston.target
    feature_names = boston.feature_names
except:
    print("\nUsing California Housing dataset (Boston deprecated)")
    from sklearn.datasets import fetch_california_housing
    california = fetch_california_housing()
    x_housing = california.data
    y_housing = california.target
    feature_names = california.feature_names

# Split and normalize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

x_train_h, x_test_h, y_train_h, y_test_h = train_test_split(
    x_housing, y_housing, test_size=0.2, random_state=42
)

scaler = StandardScaler()
x_train_h = scaler.fit_transform(x_train_h)
x_test_h = scaler.transform(x_test_h)

print(f"Training samples: {x_train_h.shape[0]}")
print(f"Features: {x_train_h.shape[1]}")
print(f"Feature names: {list(feature_names)}")


# ============================================================================
# PART 3: Linear Regression
# ============================================================================

print("\n--- Linear Regression Model ---")

linear_model = Sequential([
    Dense(1, input_shape=(x_train_h.shape[1],), activation=None)
])

linear_model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

print("Training linear regression...")
history_linear = linear_model.fit(
    x_train_h, y_train_h,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

linear_loss, linear_mae = linear_model.evaluate(x_test_h, y_test_h, verbose=0)
print(f'✓ Test MSE: {linear_loss:.4f}')
print(f'✓ Test MAE: {linear_mae:.4f}')


# ============================================================================
# PART 4: Neural Network Experiments
# ============================================================================

print("\n--- Neural Network Experiments ---\n")

# Define different architectures to test
architectures = [
    {
        'name': 'Small Network',
        'layers': [64, 32],
        'activation': 'relu'
    },
    {
        'name': 'Medium Network',
        'layers': [128, 64, 32],
        'activation': 'relu'
    },
    {
        'name': 'Deep Network',
        'layers': [256, 128, 64, 32],
        'activation': 'relu'
    },
    {
        'name': 'Tanh Network',
        'layers': [128, 64, 32],
        'activation': 'tanh'
    }
]

results = []

for arch in architectures:
    print(f"{arch['name']}:")
    print(f"  Architecture: {arch['layers']}, Activation: {arch['activation']}")
    
    # Build model
    model = Sequential()
    model.add(Dense(
        arch['layers'][0],
        input_shape=(x_train_h.shape[1],),
        activation=arch['activation']
    ))
    
    for units in arch['layers'][1:]:
        model.add(Dense(units, activation=arch['activation']))
    
    model.add(Dense(1))  # Output layer
    
    # Compile
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Train
    history = model.fit(
        x_train_h, y_train_h,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
    
    # Evaluate
    test_loss, test_mae = model.evaluate(x_test_h, y_test_h, verbose=0)
    
    results.append({
        'name': arch['name'],
        'mse': test_loss,
        'mae': test_mae,
        'history': history
    })
    
    print(f'  ✓ Test MSE: {test_loss:.4f}, MAE: {test_mae:.4f}\n')


# ============================================================================
# Results Summary
# ============================================================================

print("="*70)
print("RESULTS SUMMARY")
print("="*70)

print("\n--- MNIST Classification ---")
print(f"Custom Layers Accuracy: {test_acc:.4f}")

print("\n--- Housing Price Prediction ---")
print(f"Linear Regression:  MSE={linear_loss:.4f}, MAE={linear_mae:.4f}")

for result in results:
    improvement = (1 - result['mse']/linear_loss) * 100
    print(f"{result['name']:20s}: MSE={result['mse']:.4f}, MAE={result['mae']:.4f} "
          f"({improvement:+.1f}% vs Linear)")

# Find best model
best_idx = np.argmin([r['mse'] for r in results])
print(f"\n✓ Best performing model: {results[best_idx]['name']}")


# ============================================================================
# Visualizations
# ============================================================================

print("\n" + "="*70)
print("Creating Visualizations")
print("="*70)

# Create comprehensive visualization
fig = plt.figure(figsize=(16, 10))

# MNIST training history
ax1 = plt.subplot(2, 3, 1)
ax1.plot(history_custom.history['accuracy'], label='Train', linewidth=2)
ax1.plot(history_custom.history['val_accuracy'], label='Validation', linewidth=2)
ax1.set_title('MNIST - Training History', fontsize=12, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Linear regression training
ax2 = plt.subplot(2, 3, 2)
ax2.plot(history_linear.history['loss'], label='Train MSE', linewidth=2)
ax2.plot(history_linear.history['val_loss'], label='Val MSE', linewidth=2)
ax2.set_title('Linear Regression - Training', fontsize=12, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('MSE')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Neural network comparison (MAE)
ax3 = plt.subplot(2, 3, 3)
names = [r['name'] for r in results]
maes = [r['mae'] for r in results]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
bars = ax3.bar(range(len(names)), maes, color=colors, alpha=0.7, edgecolor='black')
ax3.axhline(y=linear_mae, color='red', linestyle='--', linewidth=2, label='Linear Baseline')
ax3.set_xticks(range(len(names)))
ax3.set_xticklabels(names, rotation=45, ha='right')
ax3.set_title('Model Comparison (MAE)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Mean Absolute Error')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Best model training history
ax4 = plt.subplot(2, 3, 4)
best_history = results[best_idx]['history']
ax4.plot(best_history.history['loss'], label='Train MSE', linewidth=2)
ax4.plot(best_history.history['val_loss'], label='Val MSE', linewidth=2)
ax4.set_title(f'Best Model ({results[best_idx]["name"]}) - Training', 
              fontsize=12, fontweight='bold')
ax4.set_xlabel('Epoch')
ax4.set_ylabel('MSE')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Model comparison (MSE)
ax5 = plt.subplot(2, 3, 5)
mses = [r['mse'] for r in results]
bars = ax5.bar(range(len(names)), mses, color=colors, alpha=0.7, edgecolor='black')
ax5.axhline(y=linear_loss, color='red', linestyle='--', linewidth=2, label='Linear Baseline')
ax5.set_xticks(range(len(names)))
ax5.set_xticklabels(names, rotation=45, ha='right')
ax5.set_title('Model Comparison (MSE)', fontsize=12, fontweight='bold')
ax5.set_ylabel('Mean Squared Error')
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

# Improvement percentages
ax6 = plt.subplot(2, 3, 6)
improvements = [(1 - r['mse']/linear_loss) * 100 for r in results]
bars = ax6.barh(range(len(names)), improvements, color=colors, alpha=0.7, edgecolor='black')
ax6.set_yticks(range(len(names)))
ax6.set_yticklabels(names)
ax6.set_title('Improvement over Linear Regression', fontsize=12, fontweight='bold')
ax6.set_xlabel('Improvement (%)')
ax6.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax6.grid(True, alpha=0.3, axis='x')

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, improvements)):
    ax6.text(val + 1, i, f'{val:.1f}%', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
print("✓ Saved training_results.png")


# Predictions vs Actual (for best model)
# Rebuild and predict with best model
best_arch = architectures[best_idx]
final_model = Sequential()
final_model.add(Dense(
    best_arch['layers'][0],
    input_shape=(x_train_h.shape[1],),
    activation=best_arch['activation']
))
for units in best_arch['layers'][1:]:
    final_model.add(Dense(units, activation=best_arch['activation']))
final_model.add(Dense(1))

final_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
final_model.fit(x_train_h, y_train_h, epochs=100, batch_size=32, 
                validation_split=0.2, verbose=0)

y_pred = final_model.predict(x_test_h, verbose=0).flatten()

plt.figure(figsize=(10, 8))
plt.scatter(y_test_h, y_pred, alpha=0.5, s=40, edgecolors='black', linewidth=0.5)
plt.plot([y_test_h.min(), y_test_h.max()],
         [y_test_h.min(), y_test_h.max()],
         'r--', lw=3, label='Perfect Prediction')
plt.xlabel('Actual Price', fontsize=12)
plt.ylabel('Predicted Price', fontsize=12)
plt.title(f'Predictions vs Actual - {results[best_idx]["name"]}', 
          fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# Add R² score
from sklearn.metrics import r2_score
r2 = r2_score(y_test_h, y_pred)
plt.text(0.05, 0.95, f'R² = {r2:.4f}', 
         transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('predictions.png', dpi=150, bbox_inches='tight')
print("✓ Saved predictions.png")
