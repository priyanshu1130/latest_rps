#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
import os
from glob import glob


# In[16]:


train_images=[]
train_labels=[]


# In[17]:


rock=glob("C:/priyanshu's PC/project rps/data/training/rock"+"/*")
paper=glob("C:/priyanshu's PC/project rps/data/training/paper"+"/*")
scissors=glob("C:/priyanshu's PC/project rps/data/training/scissors"+"/*")


# In[18]:


len(rock)


# In[19]:


import cv2
for label,img_paths in zip([0,1,2],[rock,paper,scissors]):
    for img_path in img_paths:
        image = cv2.imread(img_path)
        image=cv2.resize(image,(256,256))
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) #also try with rgb
        train_images.append(image)
        train_labels.append(label)


# In[20]:


#cv2.imshow('rock',train_images[0])


# In[21]:


x_train=np.array(train_images)
y_train=np.array(train_labels)


# In[22]:


x_train.shape


# In[23]:


import matplotlib.pyplot as plt
import random
figure = plt.figure(figsize=(12,12))

for i in range(0,16):
    random_image = random.randint(0,len(x_train))
    figure.add_subplot(4,4,i+1)
    plt.imshow(x_train[random_image])
    plt.axis("off")
    plt.title(y_train[random_image])

plt.show()


# In[24]:


#validation data
val_images=[]
val_labels=[]

val_rock=glob("C:/priyanshu's PC/project rps/data/validation/rock"+"/*")
val_paper=glob("C:/priyanshu's PC/project rps/data/validation/paper"+"/*")
val_scissors=glob("C:/priyanshu's PC/project rps/data/validation/scissors"+"/*")

for label,img_paths in zip([0,1,2],[val_rock,val_paper,val_scissors]):
    for img_path in img_paths:
        image = cv2.imread(img_path)
        image=cv2.resize(image,(256,256))
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) #also try with rgb
        val_images.append(image)
        val_labels.append(label)
x_val=np.array(val_images)
y_val=np.array(val_labels)

x_val.shape


# In[25]:


import pandas as pd
encoded_labels = pd.get_dummies(pd.DataFrame(y_train),columns=[0])
print(encoded_labels.head())
encoded_labels = np.array(encoded_labels)


# In[26]:


from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[27]:


get_ipython().system('pip install keras-tuner')


# In[28]:


from kerastuner import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters


# In[29]:


def build_model(hp):
    model= keras.Sequential([
    keras.layers.Conv2D(
        filters=hp.Int('conv_1_filter', min_value=32, max_value=128, step=16),
        kernel_size=hp.Choice('conv_1_kernel', values = [3,5]),
        activation='relu',
        input_shape=(256,256,3)
    ),
    keras.layers.Conv2D(
        filters=hp.Int('conv_2_filter', min_value=32, max_value=64, step=16),
        kernel_size=hp.Choice('conv_2_kernel', values = [3,5]),
        activation='relu'
    ),
    keras.layers.Flatten(),
    keras.layers.Dense(
        units=hp.Int('dense_1_units', min_value=32, max_value=128, step=16),
        activation='relu'
    ),
    keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3])),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    return model


# In[30]:


tuner_search=RandomSearch(build_model,max_trials=5,
                          objective='val_accuracy',directory='output',project_name="latest_rps_with_tuner")


# In[31]:


tuner_search.search(x_train,encoded_labels,epochs=3,validation_split=0.1)


# In[32]:


print(tuner_search.get_best_hyperparameters)


# In[33]:


model=tuner_search.get_best_models(num_models=1)[0]


# In[34]:


model.summary()


# In[35]:


model.fit(x_train,encoded_labels, epochs=10, validation_split=0.1, initial_epoch=3)


# In[39]:


model.save('C:/Users/Pablo/latest rps')


# In[45]:


LABELS_INPUT = {
    0: "rock",
    1: "paper",
    2: "scissors",
    3: "noise"
}


def map_index_to_label(index):
    return LABELS_INPUT[index]


# In[47]:


tst=glob("C:/Users/Pablo/Desktop/test"+"/*")
for img in tst:
    image=cv2.imread(img)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(image)
    plt.show()
    image=cv2.resize(image,(256,256)) #also try with rgb
    y_test=model.predict_classes(np.array([image]))
    print(map_index_to_label(y_test[0]))
    
      


# In[60]:


def play_game():
    x=random.choice([0,1,2])
    image = cv2.imread("C:/Users/Pablo/Desktop/test/3XNPuMFoaFCfRCVb.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(image)
    plt.show()
    image=cv2.resize(image,(256,256)) #also try with rgb
    y_test=model.predict_classes(np.array([image]))
    #print(np.argmax(model.predict(np.array([image])), axis=-1))
    print("your move is : ",map_index_to_label(y_test[0]))
    print("Bot move is  : ",map_index_to_label(x))
    if y_test[0]==0:
        if x==1:
            print('YOU LOOSE')
        else :
            print('YOU WON')
    elif y_test[0]==1:
        if x==2:
            print('YOU LOOSE')
        else :
            print('YOU WON')
    else :
        if x==0:
            print('YOU LOOSE')
        else :
            print('YOU WON')
            
        
      


# In[61]:


play_game()


# In[ ]:





# In[ ]:




