from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, LambdaLR, CosineAnnealingLR
from ignite.handlers.param_scheduler import CosineAnnealingScheduler
import math
import torch
import numpy as np


class MyScheduler():

    def __init__(self) -> None:
        pass


    def __new__(self, optimizer, num_epochs: int = 100, lr_policy: str = "Costant", initial_step = 30, initial_lr: float = 0.0003, step_size: int = 15, gamma: int = 0.3, stability:int = 50):

        self.last_epoch = 0
        self.step_size = step_size
        self.gamma = gamma
        self.stability = stability
        self.initial_lr = initial_lr
        self.lr_policy = lr_policy
        self.initial_step = initial_step
        
        if lr_policy == "Costant" or lr_policy == None or lr_policy == "":
            self.scheduler = StepLR(optimizer, step_size=self.step_size, gamma=1)

        if lr_policy == "Step_LR":
            self.scheduler = StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)

        elif lr_policy == "ReduceLROnPlateau":
            self.scheduler = ReduceLROnPlateau(optimizer)

        elif lr_policy == "Lambda_rule1":
            self.scheduler = LambdaLR(optimizer,  lr_lambda = self.lambda_rule1)

        elif lr_policy == "CosineAnnealingExponential":
            self.learning_rate_array = torch.tensor(CosineAnnealingScheduler.simulate_values(num_events=num_epochs, param_name='lr',
                                                                start_value=initial_lr, end_value=1e-6, cycle_size=20, start_value_mult=0.6))
            
            def lambda_rule_cos_ann_exp(epoch):
                return self.learning_rate_array[epoch][1]/initial_lr
            
            self.scheduler = LambdaLR(optimizer, lr_lambda = lambda_rule_cos_ann_exp)

        elif lr_policy == "MyCosineAnnealing":

            x1 = np.arange(0, 20, 1)    # 10 epochs of high costant learning rate (Warm part)
            x2 = np.arange(20, 80, 1)   # Central part of Annealing
            x3 = np.arange(80, 101, 1)  # Last part of low costant learning rate (Cold part)

            y1 = np.full_like(x1, initial_lr, dtype=np.double)
            y2 = (np.cos(0.25 * x2 - 17.9) + 1.08) / ( x2 - 6) * 0.002
            y3 = np.full_like(x3, 0.000008, dtype=np.double)

            self.learning_rate_array = torch.tensor(np.concatenate((y1, y2, y3)))
            
            # nested function
            def lambda_rule_cos_ann_exp(epoch):
                return self.learning_rate_array[epoch]/initial_lr
            
            self.scheduler = LambdaLR(optimizer, lr_lambda = lambda_rule_cos_ann_exp)

        elif lr_policy == "CosineAnnealing":
            self.scheduler = CosineAnnealingLR(optimizer, eta_min = 0.000001, T_max=30)

        return self.scheduler


    def get_scheduler(self):

        return self.scheduler


    def set_epoch(self, epoch):

        self.last_epoch = epoch
        

    def step(self, epoch = None, val_loss = None):

        if self.lr_policy == "ReduceLROnPlateau":
            self.scheduler.step(val_loss)
        else:
            self.scheduler.step()


    def lambda_rule1(self, epoch):
            
            if epoch <= self.initial_step:
                lr_l = self.initial_lr
            elif epoch % self.step_size == 0 and epoch < self.stability and epoch > self.initial_step:
                lr_l = self.initial_lr / (epoch / self.step_size)
            elif epoch >= self.stability:
                lr_l = self.initial_lr / (self.stability / self.step_size)
            return lr_l