import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = torchvision.datasets.EMNIST(root="./data", split='letters', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.EMNIST(root="./data", split='letters', train=False, download=True, transform=transform)

x_train = np.array([np.array(img[0]) for img in train_dataset])[:, 0, :, :]
y_train = np.array([img[1] for img in train_dataset]) - 1
x_test = np.array([np.array(img[0]) for img in test_dataset])[:, 0, :, :]
y_test = np.array([img[1] for img in test_dataset]) - 1

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

x_train = x_train / 255.0
x_test = x_test / 255.0

y_train = to_categorical(y_train, 26)
y_test = to_categorical(y_test, 26)

datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1)

model = Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(512, activation='relu', kernel_initializer='he_normal'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(256, activation='relu', kernel_initializer='he_normal'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu', kernel_initializer='he_normal'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(26, activation='softmax')
])

learning_rate = 1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(datagen.flow(x_train, y_train, batch_size=64), epochs=20, validation_data=(x_test, y_test))

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('Model Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.title('Model Loss')

plt.show()

predictions = model.predict(x_test)
letter_labels = [chr(i) for i in range(65, 91)]

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_test[i].reshape(28, 28), cmap=plt.cm.binary)
    true_label = letter_labels[np.argmax(y_test[i])]
    predicted_label = letter_labels[np.argmax(predictions[i])]
    plt.xlabel(f"True: {true_label}, Pred: {predicted_label}")
plt.show()
