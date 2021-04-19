#encoding=utf-8
import h5py
import numpy as np
import os,random
from keras.layers import Input,Reshape,ZeroPadding2D,Conv2D,Dropout,Flatten,Dense,Activation,MaxPooling2D,AlphaDropout,LSTM,TimeDistributed,Bidirectional
from keras import layers
import keras.models as Model
from keras.regularizers import *
import seaborn as sns
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import gc
from data import *

os.environ["KERAS_BACKEND"] = "tensorflow"
#%%

'''
for i in range(0,2): #读取数据
    filename = '../Data_h5/part'+str(i) + '.h5'
    print(filename)
    f = h5py.File(filename,'r')
    X_data = f['X'][:]
    Y_data = f['Y'][:]
    Z_data = f['Z'][:]
    .close()
#分割训练集测试集（70%训练 30%测试）
    n_examples = X_data.shape[0]
    n_train = int(n_examples * 0.7)
    train_idx = np.random.choice(range(0,n_examples), size=n_train, replace=False)#随机选取训练样本下标
    test_idx = list(set(range(0,n_examples))-set(train_idx))        #测试样本下标
    if i == 0:
        X_train = X_data[train_idx]
        Y_train = Y_data[train_idx]
        Z_train = Z_data[train_idx]
        X_test = X_data[test_idx]
        Y_test = Y_data[test_idx]
        Z_test = Z_data[test_idx]
    else:
        X_train = np.vstack((X_train, X_data[train_idx]))
        Y_train = np.vstack((Y_train, Y_data[train_idx]))
        Z_train = np.vstack((Z_train, Z_data[train_idx]))
        X_test = np.vstack((X_test, X_data[test_idx]))
        Y_test = np.vstack((Y_test, Y_data[test_idx]))
        Z_test = np.vstack((Z_test, Z_data[test_idx]))
'''
X_train, X_test, Y_train, Y_test, Z_train, Z_test = gen_data_uniform("../RML/RML2016.10a/RML2016.10a_dict.pkl")
print('训练集X维度：',X_train.shape)
print('训练集Y维度：',Y_train.shape)
print('训练集Z维度：',Z_train.shape)
print('测试集X维度：',X_test.shape)
print('测试集Y维度：',Y_test.shape)
print('测试集Z维度：',Z_test.shape)
#保存训练测试集
np.save("../Data_h5/testdata/X_test_v1.npy",X_test)
np.save("../Data_h5/testdata/Y_test_v1.npy",Y_test)
np.save("../Data_h5/testdata/Z_test_v1.npy",Z_test)
np.save("../Data_h5/testdata/X_train_v1.npy",X_train)
np.save("../Data_h5/testdata/Y_train_v1.npy",Y_train)
np.save("../Data_h5/testdata/Z_train_v1.npy",Z_train)



##查看数据是否正常
sample_idx = 52032 #随机下标
print('snr:',Z_train[sample_idx])
print('Y',Y_train[sample_idx])
plt_data = X_train[sample_idx].T
plt.figure(figsize=(15,5))
plt.plot(plt_data[0])
plt.plot(plt_data[1],color = 'red')
plt.show()

#%%
"""建立模型"""
classes = ['8PSK',
 'AM-DSB',
 'AM-SSB',
 'BPSK',
 'CPFSK',
 'GFSK',
 'PAM4',
 'QAM16',
 'QAM64',
 'QPSK',
 'WBFM']

data_format = 'channels_first'

def residual_stack(Xm,kennel_size,Seq,pool_size):
    #1*1 Conv Linear
    Xm = Conv2D(32, (1, 1), padding='same', name=Seq+"_conv1", kernel_initializer='glorot_normal',data_format=data_format)(Xm)
    #Residual Unit 1#keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True,
    Xm_shortcut = Xm
    Xm = Conv2D(32, kennel_size, padding='same',activation="relu",name=Seq+"_conv2", kernel_initializer='glorot_normal',data_format=data_format)(Xm)
    Xm = Conv2D(32, kennel_size, padding='same', name=Seq+"_conv3", kernel_initializer='glorot_normal',data_format=data_format)(Xm)
    Xm = layers.add([Xm,Xm_shortcut])
    Xm = Activation("relu")(Xm)
    #Residual Unit 2
    Xm_shortcut = Xm
    Xm = Conv2D(32, kennel_size, padding='same',activation="relu",name=Seq+"_conv4", kernel_initializer='glorot_normal',data_format=data_format)(Xm)
    Xm = Conv2D(32, kennel_size, padding='same', name=Seq+"_conv5", kernel_initializer='glorot_normal',data_format=data_format)(Xm)
    Xm = layers.add([Xm,Xm_shortcut])
    Xm = Activation("relu")(Xm)
    #MaxPooling
    Xm = MaxPooling2D(pool_size=pool_size, strides=pool_size, padding='valid', data_format=data_format)(Xm)
    return Xm


in_shp = X_train.shape[1:]   #每个样本的维度[1024,2]
#input layer
Xm_input = Input(in_shp)
Xm = Reshape([1,128,2], input_shape=in_shp)(Xm_input) # 4.19 1024->128

#Residual Srack
Xm = residual_stack(Xm,kennel_size=(3,2),Seq="ReStk0",pool_size=(2,2))   #shape:(512,1,32)
Xm = residual_stack(Xm,kennel_size=(3,1),Seq="ReStk1",pool_size=(2,1))   #shape:(256,1,32)
Xm = residual_stack(Xm,kennel_size=(3,1),Seq="ReStk2",pool_size=(2,1))   #shape:(128,1,32)
Xm = residual_stack(Xm,kennel_size=(3,1),Seq="ReStk3",pool_size=(2,1))   #shape:(64,1,32)
Xm = residual_stack(Xm,kennel_size=(3,1),Seq="ReStk4",pool_size=(2,1))   #shape:(32,1,32)
Xm = residual_stack(Xm,kennel_size=(3,1),Seq="ReStk5",pool_size=(2,1))   #shape:(16,1,32)

#LSTM1
Xm = TimeDistributed(Flatten(data_format=data_format))(Xm) # 4.19 remove TimeDistributed
Xm = LSTM(16,return_sequences = True,name='LSTM1')(Xm)
#LSTM2
Xm = LSTM(16,name='LSTM2')(Xm)

#Full Con1
Xm = Dense(128, activation='selu', kernel_initializer='glorot_normal', name="dense1")(Xm)
Xm = AlphaDropout(0.3)(Xm)
#Full Con2
Xm = Dense(len(classes), kernel_initializer='glorot_normal', name="dense2")(Xm)
#SoftMax
Xm = Activation('softmax')(Xm)
#Create Model
model = Model.Model(inputs=Xm_input,outputs=Xm)
adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer=adam)
model.summary()
#%%

"""训练模型"""
#############################################################################
#      当val_loss连续10次迭代不再减小或总迭代次数大于100时停止
#      将最小验证损失的模型保存
#############################################################################
print(tf.test.gpu_device_name())
filepath = '../Data_h5/modelResNet_Model_full_v1.h5'
history = model.fit(X_train,
    Y_train,
    batch_size=128,#batch_size=512
    epochs=100,
    verbose=2,
    validation_data=(X_test, Y_test),
    #validation_split = 0.3,
    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
    ])

# we re-load the best weights once training is finished
model.load_weights(filepath)

val_loss_list = history.history['val_loss']
loss_list = history.history['loss']
plt.plot(range(len(loss_list)),val_loss_list)
plt.plot(range(len(loss_list)),loss_list)
plt.show()

#%%
model.save(filepath)
