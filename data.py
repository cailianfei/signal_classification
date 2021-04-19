#encoding=utf-8
import pickle
import numpy as np

def gen_data_uniform(file_path):
    """
	   均匀的划分训练集和测试集，对每一个调制类型的每一个信噪比值，选取其中70%作为训练集，30%作为测试集
	   Args:
	   file_path:RML2016.10a的文件位置
    """
	
    # 载入文件,Xd为一个字典
    Xd = pickle.load(open(file_path, 'rb'), encoding='bytes')
    # 获取信噪比snrs和调制类型mods
    snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])	
    # 记录训练集
    x_train = []
    # 记录测试集
    x_test = []
    # 记录训练集的标签
    y_train = []
    # 记录测试集的标签
    y_test = []
    # 记录训练集中每个样本的调制类型和信噪比值
    lbl_train = []
    # 记录测试集中每个样本的调制类型和信噪比值
    lbl_test = []

    # 构造一个函数用于创建one-hot向量标签
    def one_hot(type, num):
        yy1 = np.zeros([num, len(mods)])
        yy1[np.arange(num), type] = 1
        return yy1
	
    # 对于每一个调制类型
    for i, mod in enumerate(mods):
        # 对于每一信噪比值
        for j, snr in enumerate(snrs):
            # 固定随机种子，每次生成的训练集和测试集都是一样的
            np.random.seed(2016) 
            # 获得该调制类型mod下的该信噪比值snr下的数据
            x = Xd[(mod, snr)]
            # 计算该条件下有多少个样本(都是1000个)
            n_examples = x.shape[0]
            # 取其中70%作为训练集
            n_train = int(n_examples * 0.7)
            # 从这1000个种随机选择700个
            train_idx = np.random.choice(range(0, n_examples), size=n_train, replace=False)
            # 剩余的为测试集
            test_idx = list(set(range(0, n_examples)) - set(train_idx))
            # 加入x_train
            x_train.append(x[train_idx].transpose(0,2,1))
            # 加入x_test
            x_test.append(x[test_idx].transpose(0,2,1))
            # 记录每一个样本的调制类型和信噪比值
            for k in range(n_train):
                lbl_train.append(snr)
                if k < n_examples - n_train:
                    lbl_test.append(snr)
                    # 获取训练集和测试集的标签
            y_train.append(one_hot([mods.index(mod)], n_train))
            y_test.append(one_hot([mods.index(mod)], (n_examples - n_train)))
    x_train = np.vstack(x_train)
    x_test = np.vstack(x_test)
    y_train = np.vstack(y_train)
    y_test = np.vstack(y_test)
    z_train = np.vstack(lbl_train)
    z_test = np.vstack(lbl_test)
    return x_train, x_test, y_train, y_test, z_train, z_test
