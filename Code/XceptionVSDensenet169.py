import tensorflow as tf
from tensorflow.keras.applications import Xception, DenseNet169
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Preprocess the data
num_classes = 10
x_train = tf.image.resize(x_train, (224, 224)) / 255.0
x_test = tf.image.resize(x_test, (224, 224)) / 255.0
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Function to build model
def build_model(base_model_class, input_shape=(224, 224, 3)):
    base_model = base_model_class(include_top=False, weights='imagenet', input_shape=input_shape)
    base_model.trainable = False  # Freeze base

    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Build models
xception_model = build_model(Xception)
densenet_model = build_model(DenseNet169)

# Train models (quick training for demo)
print("Training Xception...")
xception_model.fit(x_train, y_train, epochs=3, batch_size=64, validation_split=0.1, verbose=2)

print("\nTraining DenseNet169...")
densenet_model.fit(x_train, y_train, epochs=3, batch_size=64, validation_split=0.1, verbose=2)

# Evaluate models
xception_acc = xception_model.evaluate(x_test, y_test, verbose=0)[1]
densenet_acc = densenet_model.evaluate(x_test, y_test, verbose=0)[1]

# Show accuracy comparison
print(f"\n🔍 Model Accuracy Comparison on CIFAR-10:")
print(f"✅ Xception Accuracy:    {xception_acc * 100:.2f}%")
print(f"✅ DenseNet169 Accuracy: {densenet_acc * 100:.2f}%")
