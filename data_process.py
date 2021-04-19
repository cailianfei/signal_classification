import numpy as np
import os,random
import keras.models as Model
import matplotlib.pyplot as plt

plt.rcParams.update({'figure.max_open_warning': 0})
os.environ["KERAS_BACKEND"] = "tensorflow"

classes = ['OOK',
 '4ASK',
 '8ASK',
 'BPSK',
 'QPSK',
 '8PSK',
 '16PSK',
 '32PSK',
 '16APSK',
 '32APSK',
 '64APSK',
 '128APSK',
 '16QAM',
 '32QAM',
 '64QAM',
 '128QAM',
 '256QAM',
 'AM-SSB-WC',
 'AM-SSB-SC',
 'AM-DSB-WC',
 'AM-DSB-SC',
 'FM',
 'GMSK',
 'OQPSK']
filepath = 'D:/Data_h5/testdata/modelResNet_Model_full.h5'
model = Model.load_model(filepath)
model.summary()

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_correct_classification_probability(cm,SNRs,classes):
    plt.figure(figsize=(10,10))
    for i in range(0,len(classes)):
        if i<10:
            plt.plot(SNRs,cm[:,i,i],marker='o',label = classes[i])
        elif i >=20:
            plt.plot(SNRs, cm[:, i, i], marker='*', label=classes[i])
        else:
            plt.plot(SNRs, cm[:, i, i], marker='.', label=classes[i])
    plt.xlabel('SNR')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()


X_test = np.load("D:/Data_h5/testdata/X_test.npy")
Y_test = np.load("D:/Data_h5/testdata/Y_test.npy")
Z_test = np.load("D:/Data_h5/testdata/Z_test.npy")

batch_size = 256
test_Y_hat = model.predict(X_test, batch_size=256)
#绘制总体混淆矩阵
conf = np.zeros([len(classes),len(classes)])
confnorm = np.zeros([len(classes),len(classes)])
for i in range(0,X_test.shape[0]):
    j = list(Y_test[i,:]).index(1)
    k = int(np.argmax(test_Y_hat[i,:]))
    conf[j,k] = conf[j,k] + 1
for i in range(0,len(classes)):
    confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
plot_confusion_matrix(confnorm, labels=classes)

for i in range(len(confnorm)):
    print(classes[i],confnorm[i,i])
#绘制不同SNR下的混淆矩阵和不同类别的分类精度
acc = {}
Z_test = Z_test.reshape((len(Z_test)))
SNRs = np.unique(Z_test)
conf = np.zeros([len(SNRs), len(classes), len(classes)])
confnorm = np.zeros([len(SNRs), len(classes), len(classes)])
for snr in SNRs:
    X_test_snr = X_test[Z_test == snr]
    Y_test_snr = Y_test[Z_test == snr]

    pre_Y_test = model.predict(X_test_snr)

    for i in range(0, X_test_snr.shape[0]):  # 该信噪比下测试数据量
        j = list(Y_test_snr[i, :]).index(1)  # 正确类别下标
        j = classes.index(classes[j])
        k = int(np.argmax(pre_Y_test[i, :]))  # 预测类别下标
        k = classes.index(classes[k])
        conf[int((snr+20)/2),j, k] = conf[int((snr+20)/2),j, k] + 1
    for i in range(0, len(classes)):
        confnorm[int((snr+20)/2),i, :] = conf[int((snr+20)/2),i, :] / np.sum(conf[int((snr+20)/2),i, :])

    plt.figure()
    plot_confusion_matrix(confnorm[int((snr+20)/2),:,:], labels=classes, title="ConvNet Confusion Matrix (SNR=%d)" % (snr))

    cor = np.sum(np.diag(conf[int((snr+20)/2),:,:]))
    ncor = np.sum(conf[int((snr+20)/2),:,:]) - cor
    print("Overall Accuracy %s: " % snr, cor / (cor + ncor))
    acc[snr] = 1.0 * cor / (cor + ncor)
plot_correct_classification_probability(confnorm,SNRs,classes)
#绘制总体分类率
plt.plot(list(acc.keys()),list(acc.values()),marker='*')
plt.ylabel('ACC')
plt.xlabel('SNR')
plt.show()