"""
@author: jpzxshi
"""
import math
import torch
import learner as ln
    
class MIONet_periodic(ln.nn.Map):
    '''Multiple-input operator network (periodic).
    '''
    def __init__(self, sizes, activation='relu', initializer='default'):
        super(MIONet_periodic, self).__init__()
        self.sizes = sizes
        self.activation = activation
        self.initializer = initializer

        self.periodic = []
        self.ms = self.__init_modules()
        self.ps = self.__init_parameters()
        
    def forward(self, x):
        y = [a for a in x]
        for i in self.periodic:
            y[i] = 2 * math.pi * y[i]
            y[i] = torch.hstack((torch.cos(y[i]), torch.sin(y[i]), torch.cos(2 * y[i]), torch.sin(2 * y[i])))
        y = torch.stack([self.ms['Net{}'.format(i + 1)](y[i]) for i in range(len(self.sizes))])
        return torch.sum(torch.prod(y, dim=0), dim=-1, keepdim=True) + self.ps['bias']
    
    def __init_modules(self):
        modules = torch.nn.ModuleDict()
        for i in range(len(self.sizes)):
            size = self.sizes[i]
            if size[0] == 'p':
                size = [4] + size[1:]
                self.periodic.append(i)
            modules['Net{}'.format(i + 1)] = ln.nn.FNN(size, self.activation, self.initializer)
        return modules
    
    def __init_parameters(self):
        parameters = torch.nn.ParameterDict()
        parameters['bias'] = torch.nn.Parameter(torch.zeros([1]))
        return parameters