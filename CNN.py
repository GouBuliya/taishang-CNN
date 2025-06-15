# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 17:09:16 2024

@author: edree
"""
import numpy as np
import splitfolders
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalMaxPooling2D
from keras import regularizers
from keras.optimizers import Adam
import tensorflow as tf
import json

# 加载配置文件
with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

gpus = tf.config.list_physical_devices('GPU')
# if gpus: 
#     for gpu in gpus:
#           tf.config.experimental.set_memory_growth(gpu, True)
if gpus: 
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=config['gpu']['memory_limit_mb'])]
    )

logical_gpus = tf.config.list_logical_devices('GPU')
print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
input_folder = config['training']['input_folder']
output_folder = config['training']['output_folder']

#Split with a ratio of Train:Val:Test = 60%:20%:20%
splitfolders.ratio(input_folder, output=output_folder, seed=config['training']['random_seed'], 
                   ratio=tuple(config['training']['train_val_test_ratio']), group_prefix=None)  # Default values

batch_size = config['training']['batch_size']
img_height = config['image_processing']['target_size'][0]
img_width = config['image_processing']['target_size'][1]

train_datagen = ImageDataGenerator(rescale=1./255,
                               # rotation_range=40,
                               # width_shift_range=0.2,
                                #height_shift_range=0.2,
                                shear_range=config['training']['data_augmentation']['shear_range'],
                                zoom_range=config['training']['data_augmentation']['zoom_range'])

test_val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    output_folder + 'train/',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    interpolation=config['image_processing']['interpolation'])

validation_generator = test_val_datagen.flow_from_directory(
    output_folder + 'val/',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    interpolation=config['image_processing']['interpolation'],
    shuffle=False)

test_generator = test_val_datagen.flow_from_directory(
    output_folder + 'test/',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    interpolation=config['image_processing']['interpolation'],
    shuffle=False)

model = Sequential([
    Input(shape=(img_height, img_width, 3)),
    Conv2D(config['model_architecture']['conv_filters'][0], tuple(config['model_architecture']['conv_kernel_size']), 
           activation=config['model_architecture']['activation']), 
    MaxPooling2D(tuple(config['model_architecture']['pool_size'])),
    Conv2D(config['model_architecture']['conv_filters'][1], tuple(config['model_architecture']['conv_kernel_size']), 
           activation=config['model_architecture']['activation']),
    MaxPooling2D(tuple(config['model_architecture']['pool_size'])),
    Conv2D(config['model_architecture']['conv_filters'][2], tuple(config['model_architecture']['conv_kernel_size']), 
           activation=config['model_architecture']['activation']),
    MaxPooling2D(),
    Flatten(),
    Dense(config['model_architecture']['dense_units'], activation=config['model_architecture']['activation']),
    Dropout(config['training']['dropout_rate']),
    Dense(1, activation=config['model_architecture']['output_activation'])
])
model.summary()

model.compile(optimizer=Adam(learning_rate=config['training']['learning_rate']), 
              loss=config['model_architecture']['loss'], 
              metrics=config['model_architecture']['metrics'])

history = model.fit(
    train_generator,
    epochs=config['training']['epochs'],
    validation_data=validation_generator)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

eval_result = model.evaluate(test_generator)
print(f"Test Loss: {eval_result[0]}, Test Accuracy: {eval_result[1]}")



test_generator.reset()
predictions = model.predict(test_generator)
predicted_classes = np.where(predictions > config['model']['prediction_threshold'], 1, 0)
true_classes = test_generator.classes
true_classes = true_classes[:len(predicted_classes)]

print(confusion_matrix(true_classes, predicted_classes))
print(classification_report(true_classes, predicted_classes, digits=3))

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Calculate the ROC curve points
fpr, tpr, thresholds = roc_curve(true_classes, predictions)

# Calculate the AUC (Area under the ROC Curve)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
model.save(config['model']['backup_model_file'])