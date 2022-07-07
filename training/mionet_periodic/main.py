"""
@author: jpzxshi
"""
import learner as ln
from data import ADVDData
from mionet_periodic import MIONet_periodic
from postprocessing import L2_relative_error

# advection-diffusion system

def main():
    device = 'gpu' # 'cpu', 'gpu'
    #### data
    sensors1 = 100
    sensors2 = 100
    mesh = [100, 100]
    p = 100
    length_scale = 0.5
    train_num = 1000
    test_num = 1000
    ##### net
    net_type = 'MIONet_periodic' # 'MIONet', 'MIONet_periodic', 'DeepONet'
    if net_type == 'MIONet':
        sizes = [
            [sensors1, 300, 300, 300],
            [sensors2, -300],
            [2, 300, 300, 300]
            ]
        activation = 'relu'
        initializer = 'Glorot normal'
    elif net_type == 'MIONet_periodic':
        sizes = [
            [sensors1, 248, 248, 248],
            [sensors2, -248],
            ['p', 248, 248, 248],
            [1, 248, 248, 248]
            ]
        activation = 'relu'
        initializer = 'Glorot normal'
    elif net_type == 'DeepONet':
        sizes = [
            [sensors1 + sensors2, 300, 300, 300],
            [2, 300, 300, 0]
            ]
        activation = 'relu'
        initializer = 'Glorot normal'
    ##### training
    lr = 0.0002
    iterations = 100000
    batch_size = None
    print_every = 1000
    
    data = ADVDData(sensors1, sensors2, mesh, p, length_scale, train_num, test_num)
    if net_type == 'MIONet_periodic':
        data.trans_to_P()
    elif net_type == 'DeepONet':
        data.trans_to_D()
    Net_class = MIONet_periodic if net_type == 'MIONet_periodic' else ln.nn.MIONet
    net = Net_class(sizes, activation, initializer)
    
    args = {
        'data': data,
        'net': net,
        'criterion': 'MSE',
        'optimizer': 'adam',
        'lr': lr,
        'iterations': iterations,
        'batch_size': batch_size,
        'print_every': print_every,
        'save': True,
        'callback': None,
        'dtype': 'float',
        'device': device
    }
    
    ln.Brain.Init(**args)
    ln.Brain.Run()
    ln.Brain.Restore()
    ln.Brain.Output()
    
    print('L2 relative error:', L2_relative_error(data, ln.Brain.Best_model()))

if __name__ == '__main__':
    main()