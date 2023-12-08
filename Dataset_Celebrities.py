import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
from numpy import argmax
import pandas as pd


image_dir=r'C:\Users\91813\Downloads\Dataset_Celebrities\cropped'
lionel_messi=os.listdir(image_dir+ '/lionel_messi')
maria_sharapova=os.listdir(image_dir+ '/maria_sharapova')
roger_federer=os.listdir(image_dir+ '/roger_federer')
serena_williams=os.listdir(image_dir+ '/serena_williams')
virat_kohli=os.listdir(image_dir+ '/virat_kohli')

print("--------------------------------------\n")

print('The number of images of lionel_messi',len(lionel_messi))
print('The number of images of maria_sharapova',len(maria_sharapova))
print('The number of images of roger_federer',len(roger_federer))
print('The number of images of serena_williams',len(serena_williams))
print('The number of images of virat_kohli',len(virat_kohli))

print("--------------------------------------\n")


dataset=[]
label=[]
img_siz=(128,128)

for i , image_name in tqdm(enumerate(lionel_messi),desc="lionel_messi"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/lionel_messi/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(0)
        
        
for i ,image_name in tqdm(enumerate(maria_sharapova),desc="maria_sharapova"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/maria_sharapova/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(1)


for i ,image_name in tqdm(enumerate(roger_federer),desc="roger_federer"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/roger_federer/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(2)


for i ,image_name in tqdm(enumerate(serena_williams),desc="serena_williams"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/serena_williams/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(3)


for i ,image_name in tqdm(enumerate(virat_kohli),desc="virat_kohli"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/virat_kohli/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(4)       
        
        
dataset=np.array(dataset)
label = np.array(label)

print("--------------------------------------\n")
print('Dataset Length: ',len(dataset))
print('Label Length: ',len(label))
print("--------------------------------------\n")


print("--------------------------------------\n")
print("Train-Test Split")
x_train,x_test,y_train,y_test=train_test_split(dataset,label,test_size=0.2,random_state=42)

# Apply data augmentation to the training dataset
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen.fit(x_train)

print("--------------------------------------\n")

print("--------------------------------------\n")
print("Normalaising the Dataset. \n")

x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)
shape = x_train.shape[1:]

print("--------------------------------------\n")

model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape= shape))
model.add(tf.keras.layers.MaxPool2D((2,2)))

model.add(tf.keras.layers.Conv2D(48, (3,3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D((2,2)))

model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
print("--------------------------------------\n")

model.summary()


model.compile(optimizer='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics= ['accuracy'])

# Learning rate scheduling
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler)

print("--------------------------------------\n")
print("Training Started.\n")
history = model.fit(x_train, y_train, epochs=200, batch_size = 128, validation_split = 0.2)
print("Training Finished.\n")
print("--------------------------------------\n")

# Plot and save accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.savefig(r'C:\Users\91813\Downloads\Dataset_Celebrities\results\Dataset_Celebrities_accuracy_plot.png')

# Clear the previous plot
plt.clf()

# Plot and save loss
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.savefig(r'C:\Users\91813\Downloads\Dataset_Celebrities\results\Dataset_Celebrities_loss_plot.png')


print("--------------------------------------\n")
print("Model Evalutaion Phase.\n")
loss,accuracy= model.evaluate(x_test, y_test)
print(f'Accuracy: {round(accuracy*100,2)}')
print("--------------------------------------\n")


print("--------------------------------------\n")
print("Model Prediction.\n")
results = model.predict(x_test)
results = argmax(results,axis = 1)
results = pd.Series(results,name="Predicted Label")
submission = pd.concat([pd.Series(y_test,name = "Actual Label"),results],axis = 1)
submission.to_csv(r"C:\Users\91813\Downloads\Dataset_Celebrities\results\Dataset_Celebrities_CNN.csv",index=False)
print("--------------------------------------\n")