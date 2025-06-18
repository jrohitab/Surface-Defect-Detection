import tensorflow as tf
from tensorflow.keras.applications import Xception, ResNet50
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

# === CONFIG ===
image_size = (224, 224)
batch_size = 32
num_classes = len(os.listdir('dataset/train'))  # assumes one folder per class
epochs = 5  # You can increase this
train_dir = 'dataset/train'
test_dir = 'dataset/test'

# === DATA LOAD ===
datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_data = datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# === MODEL BUILDER FUNCTION ===
def build_model(base_model_class, input_shape=(224, 224, 3), num_classes=2):
    base_model = base_model_class(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze feature extractor

    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# === BUILD MODELS ===
xception_model = build_model(Xception, input_shape=image_size + (3,), num_classes=num_classes)
resnet_model = build_model(ResNet50, input_shape=image_size + (3,), num_classes=num_classes)

# === TRAIN MODELS ===
print("Training Xception...")
xception_model.fit(train_data, epochs=epochs, validation_data=test_data, verbose=2)

print("\nTraining ResNet50...")
resnet_model.fit(train_data, epochs=epochs, validation_data=test_data, verbose=2)

# === EVALUATE ===
xception_acc = xception_model.evaluate(test_data, verbose=0)[1]
resnet_acc = resnet_model.evaluate(test_data, verbose=0)[1]

# === RESULTS ===
print("\n🔍 Accuracy Comparison on Custom Dataset:")
print(f"✅ Xception Accuracy:  {xception_acc * 100:.2f}%")
print(f"✅ ResNet50 Accuracy:  {resnet_acc * 100:.2f}%")

# === PLOT ===
models = ['Xception', 'ResNet50']
accuracies = [xception_acc * 100, resnet_acc * 100]

plt.bar(models, accuracies, color=['skyblue', 'orange'])
plt.ylabel('Accuracy (%)')
plt.title('Xception vs ResNet50 on Custom Dataset')
plt.ylim(0, 100)
plt.show()
