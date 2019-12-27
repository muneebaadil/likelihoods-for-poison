import torch.optim as optim


class LGMOptimizer(optim.Optimizer):

    #def __init__(self, model_params, lgm_loss_params, lr, momentum, wd):
    def __init__(self, params, lr=-1e-5, momentum=0.9, weight_decay=0.0005):

        self.optimizer_cls = optim.SGD(model_params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.optimzer_lkd = optim.SGD(lgm_loss_params, lr=lr)

        #super(LGMOptimizer, self).__init__([model_params, lgm_loss_params], None)

    def zero_grad(self):

        self.optimizer_cls.zero_grad()
        self.optimizer_lkd.zero_grad()

    def step(self, closure=None):

        self.optimizer_cls.step()
        self.optimizer_lkd.step()

