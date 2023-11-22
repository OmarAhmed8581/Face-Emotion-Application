import numpy as np
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import gc
from keras import layers,callbacks,utils,applications,optimizers
from keras.models import Sequential,Model,load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam


files=os.listdir("./fer2013/train")
print(files)

image_array=[]  
label_array=[]
path="./fer2013/train/"
for i in range(len(files)):
    file_sub=os.listdir(path+files[i])
    if(files[i]=="neutral" or files[i]=="happy"):
        
        for k in tqdm(range(len(file_sub))):
            img=cv2.imread(path+files[i]+"/"+file_sub[k])
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            image_array.append(img)
            label_array.append(i)
    else:
        for k in tqdm(range(len(file_sub))):
            img=cv2.imread(path+files[i]+"/"+file_sub[k])
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            image_array.append(img)
            label_array.append(i)

a,b=np.unique(label_array,return_counts="True")
print(gc.collect())
image_array=np.array(image_array)/255.0
label_array=np.array(label_array)
label_to_text={0:"surprise",1:"fear",2:"angry",3:"neutral",4:"sad",5:"disgust",6:"happy"}

image_array,X_test,Y_train,Y_test=train_test_split(image_array,label_array,test_size=0.1)
print(gc.collect())

def show_examples(image,label,idx):
    fig,axes=plt.subplots(nrows=4,ncols=4,figsize=(16,16))
    for idx_f,ax in zip(idx,axes.ravel()):
        ax.imshow(image[idx_f].squeeze(),cmap="gray")
        ax.set_title(label_to_text[label[idx_f]])
    plt.show()
idx=np.random.choice(16,16)
show_examples(image_array,Y_train,idx)

model=Sequential()
pretrained_model=applications.MobileNetV2(input_shape=(48,48,3),include_top=False,weights="imagenet")
pretrained_model.trainable=True
model.add(pretrained_model)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dropout(0.3))
model.add(layers.Dense(1))
model.summary()
model.compile(optimizer=Adam(0.0001),loss="mean_squared_error",metrics=["mae"])
ckp_path="trained_model/model"
model_checkpoint=tf.keras.callbacks.ModelCheckpoint(filepath=ckp_path,
                                                   monitor="val_mae",
                                                   save_best_only=True,
                                                   save_weights_only=True,
                                                   mode="auto")
reduce_lr=tf.keras.callbacks.ReduceLROnPlateau(factor=0.9,
                                              monitor="val_mae",
                                              mode="auto",
                                              cooldown=0,
                                              patience=5,
                                              verbose=1,
                                              min_lr=1e-6)
EPOCHS=300
BATCH_SIZE=64
history=model.fit(image_array,Y_train,
                 validation_data=(X_test,Y_test),
                 batch_size=BATCH_SIZE,
                 epochs=EPOCHS,
                 callbacks=[model_checkpoint,reduce_lr])

model.load_weights(ckp_path)
prediction_val=model.predict(X_test,batch_size=BATCH_SIZE)
converter=tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model=converter.convert()
with open("model.tflite","wb") as f:
    f.write(tflite_model)
