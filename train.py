import re
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


#my file_dir
file_dir = 'D:\\DataSet\\project_material\\result\\'


f = open('atom.txt','r',encoding='UTF-8')
atom_matter = {}
for line in f.readlines():
    if(len(line)>0):
#     print(line.split()[0])
        number = re.findall(r'\d+\.?\d+?',line.split()[1])
        atom_syb = line.split()[0]
        atom_matter[atom_syb] = float(number[0])


energy_mat = []
coor_mat =[]

for file_name in os.listdir(file_dir):
    print("extracting" + file_name)
    f = open(file_dir + file_name)
    coor_sequence =[]
    for line in f.readlines():
        line = line.strip()
        if line in atom_matter:
            line = atom_matter[line]
        coor_sequence.append(float(line))
        energy = line
    coor_sequence.pop(len(coor_sequence)-1)
    if len(coor_sequence)<=1014:
        coor_sequence =coor_sequence + [0] *(1000-len(coor_sequence))
    coor_mat.append(np.array(coor_sequence))
    energy_mat.append(np.array(float(energy)))


#coor_mat and energy_mat
#now we have all the data we need!


#
X = np.asarray(coor_mat[0:5200])
Y = np.asarray(energy_mat[0:5200])


x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2)


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


model = tf.keras.models.Sequential([
#    tf.keras.layers.Embedding(1000,8,input_length = 1000,mask_zero = True),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Masking(mask_value = 0),
#     tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1000, activation='relu'),

     tf.keras.layers.Dense(2000, activation='relu'),
    tf.keras.layers.Dense(2000, activation='relu'),
    tf.keras.layers.Dense(2000, activation='relu'),
     tf.keras.layers.Dense(2000, activation='relu'),
    tf.keras.layers.Dense(1000, activation='relu'),
    tf.keras.layers.Dense(1000, activation='relu'),
       tf.keras.layers.Dense(500, activation='relu'),
    tf.keras.layers.Dense(50, activation='relu'),

  tf.keras.layers.Dense(1, activation='linear')
])


model.compile(optimizer='rmsprop',
              loss='MSE')

checkpoint_path = "training_1/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 period=10,
                                                 verbose=1)
# earlystop_callback = EarlyStopping(
#   monitor='val_accuracy', min_delta=0.0001,
#   patience=1)

with tf.device('gpu:1'):
    history = model.fit(x_train, y_train,
          epochs=100,
          validation_data=(x_test,y_test),
          callbacks=[cp_callback],
        )
# now let's see our model's proformance
model.evaluate(x_test,y_test,verbose=2)
result = model.predict(x_test)


loss_mat = [(result[i]-y_test[i])**2 for i in range(0,len(x_test))]
KL_mat =[(result[i]-y_test) for i in range(0,len(x_test))]


# plot our loss
hist = history.history
# print(type(history))
# print(history.history)
fig, ax = plt.subplots(1,2,figsize=(8,3))
ax[0].plot(hist['loss'], c='r', ls='--', label='train loss')
ax[0].plot(hist['val_loss'], c='orange', label='val loss')
ax[1].plot(loss_mat,c= 'green',label='preformance loss on test data')

for axis in ax:
    axis.set_xlabel('epoch')
    axis.legend();

plt.plot(y_test,color = 'green')
plt.plot(new_model.predict(x_test),color ='red')
model.save('model//ann_6.h5')
