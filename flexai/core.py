class Learner():
    """
    A class that handles the training of a pytorch model with the specified parameters.

    Args
    ----
        model: torch.nn.Module 
            The model that needs to be trained.
        dataloaders: Mapping [str, torch.utils.data.DataLoader]
            A dictonary containing a dataloader assigned to 'train', and optionally a 'valid' dataloaders.
        optimizer: Any
            A proper pytorch or pytorch like optimizer used in the model training.
        metrics: Mapping [str, Callable]
            A mapping of all the metrics to that need to be calculated. must contain 'loss' key specifying 
            the loss function that the model will be trained upon.
        callbacks: list [Callback] | None
            A list of all the callbacks that needs to be called. (default: None)
        device: str
            The device that will be used during training/validation.'cpu' to specify the cpu, 'cuda' to 
            use cuda. default('cpu')
    
    """
    
    def __init__(self, model, dataloaders, optimizer, metrics, callbacks=None, device='cpu'):
        """
        Constructs a learner with the intended training attributes.
        """
        self.model = model.to(device)
        self.dataloaders = dataloaders
        self.optimizer = optimizer
        self.metrics = metrics
        self.cbs = callbacks
        self.device=device
        self._epoch = 1
        self._keep_fit = True
        self._runCBS('on_init')

    @property
    def epoch(self):
        return self._epoch
    
    def fit(self, epochs, train=True, valid=True):
        """
        Trains/Validates a model for a number of epochs.

        Parameters
        ----------
            epochs (int): The number of epochs to train/validate.
            train (bool): If true, the model will be trained on dataloaders['train'], otherwise the model will not be trained.
            valid (bool): If true, the model will be evaluated on dataloaders['valid'], otherwise the model will not be evaluated.
        """
        phases = []
        if train: phases.append('train')
        if valid: phases.append('valid')
        self._runCBS('before_fit')
        for _ in range(epochs):
            for phase in phases:
                if phase == 'train':
                    self.model.train()
                    self.optimizer.zero_grad()
                else:
                    self.model.eval()
                self._runCBS('before_epoch', phase)
                for X, y in self.dataloaders[phase]:
                    X = X.to(self.device)
                    y = y.to(self.device)
                    self._runCBS('before_batch', phase, [X, y])
                    output = self.model(X)
                    loss = self.metrics['loss'](output, y)
                    m_value = {}
                    for key, func in self.metrics.items():
                        m_value[key] = func(output, y).item()
                    if phase == 'train':
                        loss.backward()
                        self.optimizer.step()
                    self._runCBS('after_batch', phase, m_value)
                    if self._stop_fit():
                        self._runCBS('after_fit')
                        return
                self._runCBS('after_epoch', phase)
            self._epoch += 1
        self._runCBS('after_fit')
    
    def stop_fit_request(self):
        """Sends a request to the learner to stop it's fitting (train/valid) loop. Inteded to be used within callbacks."""
        self._keep_fit = False

    def _runCBS(self, name, *args):
        if self.cbs:
            for cb in self.cbs:
                if hasattr(cb, name):
                    getattr(cb, name)(self, *args)

    def _stop_fit(self):
        r_value = not self._keep_fit
        self._keep_fit = True
        return r_value