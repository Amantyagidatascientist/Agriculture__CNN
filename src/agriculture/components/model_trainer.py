import tensorflow as tf
from tensorflow.keras import models,layers
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

IMAGE_SIZE=128
BATCH_SIZE=16
EPOCHS=50
CHANNELS=3


plt.figure(figsize=(13,12))
for image_batch,label_batch in dataset.take(1):
  for i in range(12):
     ax=plt.subplot(3,4,i+1)
     plt.imshow(image_batch[i].numpy().astype("uint8"))
     plt.title(class_names[label_batch[i]])
     plt.axis("off")


def get_dataset_partitions_tf(ds,train_split=0.7,val_split=0.15,test_split=0.15,shuffle=True,shuffle_size=32):
  ds_size=len(ds)

  if shuffle:
    ds=ds.shuffle(shuffle_size,seed=12)

  train_size=int(train_split*ds_size)
  val_size=int(val_split*ds_size)

  train_ds=ds.take(train_size)
  val_ds=ds.skip(train_size).skip(val_size)
  test_ds=ds.skip(train_size).skip(val_size)

  return train_ds,val_ds,test_ds

train_ds,val_ds,test_ds=get_dataset_partitions_tf(dataset)
train_ds=train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds=val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds=test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

resize_and_rescale=tf.keras.Sequential([
    layers.Resizing(IMAGE_SIZE,IMAGE_SIZE),
    layers.Rescaling(1.0/255)
                     ])
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])

input_shape=(BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,CHANNELS)
n_classes=10
model=models.Sequential([resize_and_rescale,

                          layers.Conv2D(32,(3,3),activation='relu',input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS), padding='same'),
                          layers.BatchNormalization(),
                          layers.MaxPool2D((2,2)),

                          layers.Conv2D(64,kernel_size=(3,3),activation='relu', padding='same'),
                          layers.BatchNormalization(),
                          layers.MaxPool2D((2,2)),

                          layers.Conv2D(64,kernel_size=(3,3),activation='relu', padding='same'),
                          layers.BatchNormalization(),
                          layers.MaxPool2D((2,2)),

                          layers.Conv2D(64,kernel_size=(3,3),activation='relu', padding='same'),
                          layers.BatchNormalization(),
                          layers.MaxPool2D((2,2)),

                          layers.Conv2D(64,kernel_size=(3,3),activation='relu', padding='same'),
                          layers.BatchNormalization(),
                          layers.MaxPool2D((2,2)),
                          layers.Dropout(0.5),

                          layers.Conv2D(64,kernel_size=(3,3),activation='relu',padding='same'),
                          layers.BatchNormalization(),
                          layers.MaxPool2D((2,2)),
                          layers.Dropout(0.5),

                          layers.Flatten(),
                          layers.Dense(64,activation='relu'),
                          layers.BatchNormalization(),
                          layers.Dropout(0.5),
                          layers.Dense(n_classes,activation='softmax')

])
model.build(input_shape=input_shape)



model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy']
              )

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
]


history=model.fit(train_ds,epochs=EPOCHS,batch_size=BATCH_SIZE,verbose=1,validation_data=val_ds,callbacks=[callbacks])


scores = model.evaluate(test_ds)
