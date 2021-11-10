import torch
from argparse import ArgumentParser
from torch.utils.data import DataLoader, Dataset

from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset

import time
import numpy as np
from argparse import ArgumentParser

from loggers.exp_logger import ExperimentLogger
from .LAS_utils import mixup_data, mixup_criterion,LabelAwareSmoothing, LearnableWeightScaling
import  datasets.data_loader as stage2_utils



class Appr(Inc_Learning_Appr):
    """Class implementing the joint baseline"""

    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False, eval_on_train=False,
                 logger=None, exemplars_dataset=None, freeze_after=-1):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset)
        self.trn_datasets = []
        self.val_datasets = []
        self.freeze_after = freeze_after
        self.lws_models = torch.nn.ModuleList()

        have_exemplars = self.exemplars_dataset.max_num_exemplars + self.exemplars_dataset.max_num_exemplars_per_class
        # assert (have_exemplars == 0), 'Warning: Joint does not use exemplars. Comment this line to force it.'

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset
        
    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--freeze-after', default=-1, type=int, required=False,
                            help='Freeze model except heads after the specified task'
                                 '(-1: normal Incremental Joint Training, no freeze) (default=%(default)s)')
        return parser.parse_known_args(args)

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""
        if self.freeze_after > -1 and t >= self.freeze_after:
            self.model.freeze_all()
            for head in self.model.heads:
                for param in head.parameters():
                    param.requires_grad = True

    def _get_optimizer(self):
        """Returns the optimizer"""
        return torch.optim.SGD([{"params": self.model.model.parameters()},
                                    {"params": self.model.heads.parameters()},{"params":self.lws_models.parameters()}], 
                                    lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    def pre_train_process(self, t, trn_loader):
        lws_model = LearnableWeightScaling(num_classes=self.model.task_cls[t]).to(self.device)
        self.lws_models.append(lws_model)
        super().pre_train_process(t, trn_loader)




    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""

        # add new datasets to existing cumulative ones
        self.trn_datasets.append(trn_loader.dataset)
        self.val_datasets.append(val_loader.dataset)
        trn_dset = JointDataset(self.trn_datasets)
        val_dset = JointDataset(self.val_datasets)
        # balance_sampler = stage2_utils.ClassAwareSampler(trn_dset)
        trn_loader = DataLoader(trn_dset,
                                batch_size=trn_loader.batch_size,
                                shuffle=True,
                                num_workers=trn_loader.num_workers,
                                pin_memory=trn_loader.pin_memory)
        val_loader = DataLoader(val_dset,
                                batch_size=val_loader.batch_size,
                                shuffle=False,
                                num_workers=val_loader.num_workers,
                                pin_memory=val_loader.pin_memory)
        # balanced_trn_loader = DataLoader(trn_dset,
        #                         batch_size=trn_loader.batch_size,
        #                         shuffle=False,
        #                         num_workers=trn_loader.num_workers,
        #                         pin_memory=trn_loader.pin_memory,
        #                         sampler=balance_sampler)

        # continue training as usual
        lr = self.lr
        best_loss = np.inf
        patience = self.lr_patience
        best_model = self.model.get_copy()

        self.optimizer = self._get_optimizer()
        # Loop epochs
        for e in range(self.nepochs):
            # Train
            clock0 = time.time()
            self.train_epoch(t, trn_loader)
            clock1 = time.time()
            if self.eval_on_train:
                train_loss, train_acc, _ = self.eval(t, trn_loader)
                clock2 = time.time()
                print('| Epoch {:3d}, time={:5.1f}s/{:5.1f}s | Train: loss={:.3f}, TAw acc={:5.1f}% |'.format(
                    e + 1, clock1 - clock0, clock2 - clock1, train_loss, 100 * train_acc), end='')
                self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=train_loss, group="train")
                self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * train_acc, group="train")
            else:
                print('| Epoch {:3d}, time={:5.1f}s | Train: skip eval |'.format(e + 1, clock1 - clock0), end='')

            # Valid
            clock3 = time.time()
            valid_loss, valid_acc, _ = self.eval(t, val_loader)
            clock4 = time.time()
            print(' Valid: time={:5.1f}s loss={:.3f}, TAw acc={:5.1f}% |'.format(
                clock4 - clock3, valid_loss, 100 * valid_acc), end='')
            self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=valid_loss, group="valid")
            self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * valid_acc, group="valid")

            # Adapt learning rate - patience scheme - early stopping regularization
            if valid_loss < best_loss:
                # if the loss goes down, keep it as the best model and end line with a star ( * )
                best_loss = valid_loss
                best_model = self.model.get_copy()
                patience = self.lr_patience
                print(' *', end='')
            else:
                # if the loss does not go down, decrease patience
                patience -= 1
                if patience <= 0:
                    # if it runs out of patience, reduce the learning rate
                    lr /= self.lr_factor
                    print(' lr={:.1e}'.format(lr), end='')
                    if lr < self.lr_min:
                        # if the lr decreases below minimum, stop the training session
                        print()
                        break
                    # reset patience and recover best model so far to continue training
                    patience = self.lr_patience
                    self.optimizer.param_groups[0]['lr'] = lr
                    # self.model.set_state_dict(best_model)
            self.logger.log_scalar(task=t, iter=e + 1, name="patience", value=patience, group="train")
            self.logger.log_scalar(task=t, iter=e + 1, name="lr", value=lr, group="train")
            print()
        lr=self.lr
        for e in range(30):
        # Train
            clock0 = time.time()
            # self.train_epoch_stage2(t, balanced_trn_loader)
            self.train_epoch_stage2(t, trn_loader)
            clock1 = time.time()
            if self.eval_on_train:
                train_loss, train_acc, _ = self.eval(t, trn_loader)
                clock2 = time.time()
                print('| Epoch {:3d}, time={:5.1f}s/{:5.1f}s | Train: loss={:.3f}, TAw acc={:5.1f}% |stage2'.format(
                    e + 1, clock1 - clock0, clock2 - clock1, train_loss, 100 * train_acc), end='')
                self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=train_loss, group="train")
                self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * train_acc, group="train")
            else:
                print('| Epoch {:3d}, time={:5.1f}s | Train: skip eval |stage2'.format(e + 1, clock1 - clock0), end='')

            # Valid
            clock3 = time.time()
            valid_loss, valid_acc, _ = self.eval(t, val_loader)
            clock4 = time.time()
            print(' Valid: time={:5.1f}s loss={:.3f}, TAw acc={:5.1f}% |'.format(
                clock4 - clock3, valid_loss, 100 * valid_acc), end='')
            self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=valid_loss, group="valid")
            self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * valid_acc, group="valid")

            # Adapt learning rate - patience scheme - early stopping regularization
            if valid_loss < best_loss:
                # if the loss goes down, keep it as the best model and end line with a star ( * )
                best_loss = valid_loss
                best_model = self.model.get_copy()
                patience = 10
                print(' *', end='')
            else:
                # if the loss does not go down, decrease patience
                patience -= 1
                if patience <= 0:
                    # if it runs out of patience, reduce the learning rate
                    lr /= self.lr_factor
                    print(' lr={:.1e}'.format(lr), end='')
                    # if lr < self.lr_min:
                    #     # if the lr decreases below minimum, stop the training session
                    #     print()
                    #     break
                    # reset patience and recover best model so far to continue training
                    patience = 10
                    self.optimizer.param_groups[0]['lr'] = lr
                    # self.model.set_state_dict(best_model)
            self.logger.log_scalar(task=t, iter=e + 1, name="patience", value=patience, group="train")
            self.logger.log_scalar(task=t, iter=e + 1, name="lr", value=lr, group="train")
            print()
        # self.model.set_state_dict(best_model)

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        if self.freeze_after < 0 or t <= self.freeze_after:
            self.model.train()
            if self.fix_bn and t > 0:
                self.model.freeze_bn()
        else:
            self.model.eval()
            for head in self.model.heads:
                head.train()
        for images, targets in trn_loader:
            # Forward current model
            outputs = self.model(images.to(self.device))
            loss = self.criterion(t, outputs, targets.to(self.device))
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

    # def train_epoch(self, t , train_loader):
    #     self.model.train()
    #     training_data_num = len(train_loader.dataset)
    #     end_steps = int(training_data_num / train_loader.batch_size)
    #     for i, (images, target) in enumerate(train_loader):
    #         if i > end_steps:
    #             break
    #         images, targets_a, targets_b, lam = mixup_data(images, target, alpha=1.0)
    #         output = self.model(images.to(self.device))  
    #         loss = mixup_criterion(self.criterion, output, targets_a.to(self.device), targets_b.to(self.device), lam)
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #         torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
    #         self. optimizer.step()

    def train_epoch_stage2(self , t,train_loader):
        training_data_num = len(train_loader.dataset)
        end_steps = int(np.ceil(float(training_data_num) / float(train_loader.batch_size)))

        # switch to train mode
        # print(self.model.model)
        # print(self.model.heads)
        self.model.model.eval()
        self.model.heads.train()
        for i, (images, target) in enumerate(train_loader):
            if i > end_steps:
                break
            with torch.no_grad():
                feat = self.model.model(images.to(self.device))
            output=[]
            for idx in range(len(self.model.heads)):
                output.append(self.model.heads[idx](feat))
                # output[idx] = self.lws_models[idx](output[idx])
            # output = lws_model(output)
            loss = self.criterion(t,output, target.to(self.device))
            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

    # def train(self, t, trn_loader, val_loader):
    #     """Main train structure"""
    #     self.pre_train_process(t, trn_loader)
    #     self.train_loop(t, trn_loader, val_loader)
    #     # self.train_loop_stage_2(t,trn_loader,val_loader)
    #     self.post_train_process(t, trn_loader)
        

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""
        pass

    def criterion(self,t, outputs, targets):
        """Returns the loss value"""
        return torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)


    def train(self, t, trn_loader, val_loader):
            """Main train structure"""
            self.pre_train_process(t, trn_loader)
            self.train_loop(t, trn_loader, val_loader)
            self.post_train_process(t, trn_loader)

class JointDataset(Dataset):
    """Characterizes a dataset for PyTorch -- this dataset accumulates each task dataset incrementally"""

    def __init__(self, datasets):
        self.datasets = datasets
        self._len = sum([len(d) for d in self.datasets])
        self.targets=[]
        self.get_targets()

    def get_targets(self):
        for d in self.datasets:
            for idx in range(len(d)):
                x,y = d[idx]
                self.targets.append(y)

    def __len__(self):
        'Denotes the total number of samples'
        return self._len

    def __getitem__(self, index):
        for d in self.datasets:
            if len(d) <= index:
                index -= len(d)
            else:
                x, y = d[index]
                return x, y




