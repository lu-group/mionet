"""
@author: jpzxshi
"""
import numpy as np
import learner as ln
from data import ADVDData
from mionet_periodic import MIONet_periodic

def L2_relative_error(data, net):
    test_num = 200
    test_mesh = [100, 100]
    p = test_mesh[0] * test_mesh[1]
    data_test = ADVDData(data.sensors1, data.sensors2, test_mesh, None, data.length_scale, 1, test_num)
    E = []
    for i in range(test_num):
        D = data_test.X_test[0][i * p: (i + 1) * p]
        V = data_test.X_test[1][i * p: (i + 1) * p]
        xy = data_test.X_test[2][i * p: (i + 1) * p]
        if isinstance(net, ln.nn.MIONet) and len(net.sizes) == 2:
            inp = (np.hstack((D, V)), xy)
        elif isinstance(net, MIONet_periodic):
            inp = (D, V, xy[..., :1], xy[..., 1:])
        else:
            inp = (D, V, xy)
        value = data_test.y_test[i * p: (i + 1) * p].squeeze()
        value_pred = net.predict(inp, returnnp=True).squeeze()
        E.append(np.linalg.norm(value - value_pred) / np.linalg.norm(value))
    return np.mean(E)