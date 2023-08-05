import torch
import numpy as np
from abc import abstractmethod
from tensorboardX import SummaryWriter
from utils.torch_utils import bias_parameters, weight_parameters, other_parameters, load_checkpoint, save_checkpoint, AdamW


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, train_loaders, valid_loaders, model, loss_func,
                 _log, save_root, config, resume=False, train_sets_epoches=[np.inf]):
        self._log = _log

        self.cfg = config
        self.save_root = save_root
        self.summary_writer = SummaryWriter(str(save_root))

        self.train_loaders, self.valid_loaders = train_loaders, valid_loaders
        self.train_sets_epoches = train_sets_epoches
        self.device, self.device_ids = self._prepare_device(config['n_gpu'])

        self.loss_func = loss_func
        
            
        if not resume:
            self.model = self._init_model(model)
            self.i_epoch, self.i_iter = 0, 0
            self.i_train_set = 0
            while self.train_sets_epoches[self.i_train_set] == 0:  # skip the datasets of 0 epoches
                self.i_train_set += 1
            
            self.optimizer = self._create_optimizer()
            self.scheduler = self._create_scheduler(self.optimizer, self.train_sets_epoches[self.i_train_set])

            self.best_error = np.inf

        else:  # load all states
            self._load_resume_ckpt(model)
            
    @abstractmethod
    def _run_one_epoch(self):
        ...

    @abstractmethod
    def _validate_with_gt(self):
        ...

    def train(self):
        
        if self.cfg.pretrained_model is not None:  # if using a pretrained model, evaluate that first to compare
            errors, error_names = self._validate_with_gt()
            valid_res = ' '.join('{}: {:.2f}'.format(*t) for t in zip(error_names, errors))
            self._log.info(' * Epoch {} '.format(self.i_epoch) + valid_res)
        
            
        for epoch in range(self.i_epoch, self.cfg.epoch_num):
            self._run_one_epoch()
            
            if self.i_epoch >= sum(self.train_sets_epoches[:(self.i_train_set + 1)]):
                self.i_train_set += 1
                self.optimizer = self._create_optimizer() # reset the states of optimizer as well
                self.scheduler = self._create_scheduler(self.optimizer, self.train_sets_epoches[self.i_train_set])
                
            if self.i_epoch % self.cfg.val_epoch_size == 0:
                errors, error_names = self._validate_with_gt()
                valid_res = ' '.join(
                    '{}: {:.2f}'.format(*t) for t in zip(error_names, errors))
                self._log.info(' * Epoch {} '.format(self.i_epoch) + valid_res)

    def zero_grad(self):
        # One Pytorch tutorial suggests clearing the gradients this way for faster speed
        # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
        for param in self.model.parameters():
            param.grad = None
            
    def _init_model(self, model):
        model = model.to(self.device)
        if self.cfg.pretrained_model:
            self._log.info("=> using pre-trained weights {}.".format(
                self.cfg.pretrained_model))
            epoch, weights = load_checkpoint(self.cfg.pretrained_model)
            model.load_state_dict(weights)
        else:
            self._log.info("=> Train from scratch.")
            model.init_weights()
        model = torch.nn.DataParallel(model, device_ids=self.device_ids)

        self._log.info("number of parameters: {}".format(self.count_parameters(model)))
        self._log.info("gpu memory allocated (model parameters only): {} Bytes".format(torch.cuda.memory_allocated()))
        return model

    def _create_optimizer(self):
        self._log.info('=> setting {} optimizer'.format(self.cfg.optim))
        param_groups = [
            {'params': bias_parameters(self.model.module),
             'weight_decay': self.cfg.bias_decay},
            {'params': weight_parameters(self.model.module),
             'weight_decay': self.cfg.weight_decay},
            {'params': other_parameters(self.model.module),
             'weight_decay': 0} ]

        if self.cfg.optim == 'adamw':
            optimizer = AdamW(param_groups, self.cfg.lr,
                              betas=(self.cfg.momentum, self.cfg.beta))
        elif self.cfg.optim == 'adam':
            optimizer = torch.optim.Adam(param_groups, self.cfg.lr,
                                         betas=(self.cfg.momentum, self.cfg.beta),
                                         eps=1e-7)
        else:
            raise NotImplementedError(self.cfg.optim)

        return optimizer
    
    def _create_scheduler(self, optimizer, epoches=np.inf):
        
        if self.i_train_set < len(self.train_sets_epoches) - 1:  # try only the last loader uses onecyclelr
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)
            return scheduler
          
        if "lr_scheduler" in self.cfg.keys():
            self._log.info('=> setting {} scheduler'.format(self.cfg.lr_scheduler.module))
        
            if self.cfg.lr_scheduler.module == 'OneCycleLR':
                from utils.one_cycle_lr import OneCycleLR
                params = self.cfg.lr_scheduler.params
                params['epochs'] = min(epoches, self.cfg.epoch_num - self.i_epoch)
                params['steps_per_epoch'] = self.cfg.epoch_size
                scheduler = OneCycleLR(optimizer, **params)
            else:
                scheduler = getattr(torch.optim.lr_scheduler, self.cfg.lr_scheduler.module)(optimizer, **self.cfg.lr_scheduler.params)
        else:    # a dummy scheduler by default
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)
    
        return scheduler
    
    def _load_resume_ckpt(self, model):
        self._log.info('==> resuming')
        ckpt_dict = torch.load(self.save_root / 'model_ckpt.pth.tar')
        
        if 'iter' not in ckpt_dict.keys():
            ckpt_dict['iter'] = ckpt_dict['epoch'] * self.cfg.epoch_size
        if 'best_error' not in ckpt_dict.keys():
            ckpt_dict['best_error'] = np.inf
        self.i_epoch, self.i_iter, self.best_error = ckpt_dict['epoch'], ckpt_dict['iter'], ckpt_dict['best_error']
        self.i_train_set = np.where(self.i_epoch < np.cumsum(self.train_sets_epoches))[0][0]
        
        model = model.to(self.device)
        model.load_state_dict(ckpt_dict['state_dict'])
        self.model = torch.nn.DataParallel(model, device_ids=self.device_ids)
        
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler(self.optimizer, self.train_sets_epoches[self.i_train_set])

        if 'optimizer_dict'in ckpt_dict.keys():
            self.optimizer.load_state_dict(ckpt_dict['optimizer_dict'])
        if 'scheduler_dict'in ckpt_dict.keys():
            self.scheduler.load_state_dict(ckpt_dict['scheduler_dict'])
        
        return 
 
    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self._log.warning("Warning: There\'s no GPU available on this machine,"
                              "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self._log.warning(
                "Warning: The number of GPU\'s configured to use is {}, "
                "but only {} are available.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        self._log.info('=> gpu in use: {} gpu(s)'.format(n_gpu_use))
        self._log.info('device names: {}'.format([torch.cuda.get_device_name(i) for i in list_ids]))
        return device, list_ids

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def save_model(self, error, name, save_with_runtime=True):
        if save_with_runtime:
            models = {'epoch': self.i_epoch,
                      'iter': self.i_iter,
                      'best_error': self.best_error,
                      'state_dict': self.model.module.state_dict(),
                      'optimizer_dict': self.optimizer.state_dict(),
                      'scheduler_dict': self.scheduler.state_dict() }
        else:
            models = {'state_dict': self.model.module.state_dict()}

        save_checkpoint(self.save_root, models, name, is_best=False)
