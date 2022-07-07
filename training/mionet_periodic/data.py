"""
@author: jpzxshi
"""
import numpy as np
from scipy import interpolate
from sklearn import gaussian_process as gp
import learner as ln

def solve_ADVD(xmin, xmax, tmin, tmax, D, V, Nx, Nt):
    """Solve
    u_t + u_x - D * u_xx = 0
    u(x, 0) = V(x) periodic, twice continuously differentiable
    D(x) periodic, continuous
    """
    # Crank-Nicholson
    x = np.linspace(xmin, xmax, Nx)
    t = np.linspace(tmin, tmax, Nt)
    h = x[1] - x[0]
    dt = t[1] - t[0]
    lam = dt / h
    mu = dt / h ** 2
    u = np.zeros([Nx, Nt])
    u[:, 0] = V(x)
    d = D(x[1:])

    I = np.eye(Nx - 1)
    I1 = np.roll(I, 1, axis=0)
    I2 = np.roll(I, -1, axis=0)
    A = (1 + d * mu) * I - (lam / 4 + d * mu / 2) * I1 + (lam / 4 - d * mu / 2) * I2
    B = 2 * I - A
    C = np.linalg.solve(A, B)

    for n in range(Nt - 1):
        u[1:, n + 1] = C @ u[1:, n]
    u[0, :] = u[-1, :]

    return u

class ADVDData(ln.Data):
    '''Data for learning Advection-diffusion equation.
    u_t + u_x - D * u_xx = 0
    u(x, 0) = V(x) periodic, twice continuously differentiable
    D(x) periodic, continuous
    '''
    def __init__(self, sensors1, sensors2, mesh, p, length_scale, train_num, test_num):
        super(ADVDData, self).__init__()
        self.sensors1 = sensors1
        self.sensors2 = sensors2
        self.mesh = mesh
        self.p = p
        self.length_scale = length_scale
        self.train_num = train_num
        self.test_num = test_num
        self.__init_data()
       
    def __init_data(self):
        features = 1000
        train1, train2 = self.__gaussian_process(self.train_num, features), self.__gaussian_process(self.train_num, features)
        test1, test2 = self.__gaussian_process(self.test_num, features), self.__gaussian_process(self.test_num, features)
        self.X_train, self.y_train = self.__generate(train1, train2)
        self.X_test, self.y_test = self.__generate(test1, test2)
        
    def __generate(self, gps1, gps2):
        def generate(gp1, gp2):
            f1 = lambda x: interpolate.interp1d(np.linspace(0, 1, num=gp1.shape[-1]), gp1, kind='cubic', copy=False, assume_sorted=True)(x)
            f2 = lambda x: interpolate.interp1d(np.linspace(0, 1, num=gp2.shape[-1]), gp2, kind='cubic', copy=False, assume_sorted=True)(x)
            D = lambda x: 0.05 + 0.05 * np.abs(f1(np.sin(np.pi * x) ** 2))
            V = lambda x: f2(np.sin(np.pi * x) ** 2)
            u = solve_ADVD(0, 1, 0, 1, D, V, *self.mesh)
            if self.p is None:
                p = np.array([[i, j] for i in range(self.mesh[0]) for j in range(self.mesh[1])])
            else:
                p = np.hstack((np.random.randint(self.mesh[0], size=self.p)[:, None], np.random.randint(self.mesh[1], size=self.p)[:, None]))
            s = np.array(list(map(lambda x: u[tuple(x)], p)))[:, None]
            D_sensors = D(np.linspace(0, 1, num=self.sensors1))
            V_sensors = V(np.linspace(0, 1, num=self.sensors2))
            p = p / np.array([self.mesh[0] - 1, self.mesh[1] - 1])
            return np.hstack([np.tile(D_sensors, (p.shape[0], 1)), np.tile(V_sensors, (p.shape[0], 1)), p, s])
        res = np.vstack(list(map(generate, gps1, gps2)))
        return (res[..., :self.sensors1], res[..., self.sensors1:-3], res[..., -3:-1]), res[..., -1:]
    
    def __gaussian_process(self, num, features):
        x = np.linspace(0, 1, num=features)[:, None]
        A = gp.kernels.RBF(length_scale=self.length_scale)(x)
        L = np.linalg.cholesky(A + 1e-13 * np.eye(features))
        return (L @ np.random.randn(features, num)).transpose()
    
    def trans_to_D(self):
        if isinstance(self.y_train, np.ndarray):
            self.X_train = (np.hstack([self.X_train[0], self.X_train[1]]), self.X_train[2])
            self.X_test = (np.hstack([self.X_test[0], self.X_test[1]]), self.X_test[2])
        else:
            import torch
            self.X_train = (torch.hstack([self.X_train[0], self.X_train[1]]), self.X_train[2])
            self.X_test = (torch.hstack([self.X_test[0], self.X_test[1]]), self.X_test[2])
            
    def trans_to_P(self):
        self.X_train = (self.X_train[0], self.X_train[1], self.X_train[2][..., :1], self.X_train[2][..., 1:])
        self.X_test = (self.X_test[0], self.X_test[1], self.X_test[2][..., :1], self.X_test[2][..., 1:])