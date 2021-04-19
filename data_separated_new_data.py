import numpy as np
import h5py

f = h5py.File('../2018.01/GOLD_XYZ_OSC.0001_1024.hdf5', 'r')
dir_path = '../Data_h5'

for modu in range(24):
    X_list = []
    Y_list = []
    Z_list = []
    print('part ', modu)
    start_modu = modu * 106496
    X = f['X'][start_modu:start_modu + 106495]
    X_list.append(X)
    Y_list.append(f['Y'][start_modu:start_modu + 106495])
    Z_list.append(f['Z'][start_modu:start_modu + 106495])
    filename = dir_path + '/part' + str(modu) + '.h5'
    fw = h5py.File(filename, 'w')
    fw['X'] = np.vstack(X_list)
    fw['Y'] = np.vstack(Y_list)
    fw['Z'] = np.vstack(Z_list)
    print('X shape:', fw['X'].shape)
    print('Y shape:', fw['Y'].shape)
    print('Z shape:', fw['Z'].shape)
    fw.close()
f.close()
