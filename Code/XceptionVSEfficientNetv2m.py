import tensorflow as tf
from tensorflow.keras.applications import Xception, EfficientNetV2M
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
num_classes = 10

# Resize and normalize CIFAR-10 images to 224x224x3
x_train = tf.image.resize(x_train, (224, 224)) / 255.0
x_test = tf.image.resize(x_test, (224, 224)) / 255.0
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Function to build the model
def build_model(base_model_class, input_shape=(224, 224, 3)):
    base_model = base_model_class(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze base layers

    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Build both models
xception_model = build_model(Xception)
efficientnet_model = build_model(EfficientNetV2M)

# Train models (for quick demo: 3 epochs)
print("Training Xception...")
xception_model.fit(x_train, y_train, epochs=3, batch_size=64, validation_split=0.1, verbose=2)

print("\nTraining EfficientNetV2M...")
efficientnet_model.fit(x_train, y_train, epochs=3, batch_size=64, validation_split=0.1, verbose=2)

# Evaluate both models
xception_acc = xception_model.evaluate(x_test, y_test, verbose=0)[1]
efficientnet_acc = efficientnet_model.evaluate(x_test, y_test, verbose=0)[1]

# Print comparison
print("\n🔍 Model Accuracy Comparison on CIFAR-10:")
print(f"✅ Xception Accuracy:        {xception_acc * 100:.2f}%")
print(f"✅ EfficientNetV2M Accuracy: {efficientnet_acc * 100:.2f}%")

# Optional: Plot bar graph
models = ['Xception', 'EfficientNetV2M']
accuracies = [xception_acc * 100, efficientnet_acc * 100]

plt.bar(models, accuracies, color=['skyblue', 'lightcoral'])
plt.ylabel('Accuracy (%)')
plt.title('Model Accuracy Comparison')
plt.ylim(0, 100)
plt.show()
