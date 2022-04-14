import torch 

torch.manual_seed(0)

class AddUniformNoise(object):
    def __init__(self, min=0., max=1.):
        self.min = min
        self.max = max
        
    def __call__(self, tensor):
        return tensor + torch.rand(tensor.size()) * (self.max - self.min) + self.min
    
    def __repr__(self):
        return self.__class__.__name__ + '(min={0}, max={1})'.format(self.min, self.max)

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
