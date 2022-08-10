"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import torch
import itertools
import warnings

from abc import ABC, abstractmethod
from typing import Dict, List, Any
from collections import OrderedDict
from torch.cuda.amp import GradScaler

from speechain.model.abs import Model
from speechain.utilbox.import_util import import_class

class OptimScheduler(ABC):
    """
    OptimScheduler is the base class for all optimscheduler in this toolkit. The main job of the
    optimscheduler is optimizing the target model parameters and scheduling the learning rate during training. In
    this toolkit, we combine traditional optimizers and schedulers into a single class. Each optimeduler has one
    built-in optimizer member (torch.optim.Optimizer) which is initialized automatically by the 'optim_conf' given in
    your configuration.

    """
    def __init__(self,
                 optim_type: str,
                 optim_conf: Dict[str, Any],
                 model: Model,
                 optim_losses: str or List[str] = None,
                 updated_modules: List[str] = None,
                 step_per_update: int = 1,
                 accum_grad: int = 1,
                 ft_factor: float = 1.0,
                 grad_clip: float = 1.0,
                 grad_norm_type: float = 2.0,
                 **sche_conf):
        """

        Args:
            optim_type: str
                The optimizer query used to pick up the target torch.optim.Optimizer from optim_class_dict
            optim_conf: Dict
                The optimizer configuration used to initialize the optimizer
            model: Model
                The model to be update.
            optim_losses: str or List[str]
                The target losses used in this optimscheduler to calculate the gradients.
                The value can be either a string (only one loss) or a list (multiple losses).
                None means all the losses will be used.
            updated_modules: str or List[str]
                The target modules to be updated in this optimscheduler.
                The value can be either a string (only one module) or a list (multiple modules).
                None means the entire model will be updated.
            accum_grad: int
                The number of steps for gradient accumulation.
                Received from the runner by exp_cfg.
            step_per_update: int
                The updating interval for the built-in optimizer.
                The parameter updating will be done once every step_per_update steps.
            step_num: int
                The initial step number.
            visdom_lr: bool
                Unused
            vis: Any
                Unused
            **sche_conf: Dict
                The customized arguments to initialize the scheduler part of this optimeduler.
        """

        # initialize the general part of the scheduler
        assert accum_grad >= 1 and step_per_update >= 1, \
            f"Both of accum_grad and step_per_update should be equal to or larger than 1, " \
            f"but got accum_grad={accum_grad} and step_per_update={step_per_update}."
        self.model = model
        self.tmp_grads = None

        # gradient-related arguments (loaded from exp_cfg)
        self.accum_grad = accum_grad
        self.grad_clip = grad_clip
        self.grad_norm_type = grad_norm_type
        self.ft_factor = ft_factor

        # optimization-related arguments (loaded from train_cfg)
        self.step_per_update = step_per_update
        self.optim_losses = optim_losses if isinstance(optim_losses, (List, type(None))) else [optim_losses]
        self.updated_modules = updated_modules if isinstance(updated_modules, (List, type(None))) else [updated_modules]
        # all parameters of the model are returned
        if self.updated_modules is None:
            params = self.model.parameters()
        # specific parameters are be updated
        else:
            _updated_modules = [self.model.__getattr__(module).parameters() for module in self.updated_modules]
            params = itertools.chain(*_updated_modules)

        # initialize the optimizer part
        optim_class = import_class('torch.optim.' + optim_type)
        self.optimizer = optim_class(params=params, **optim_conf)

        # initialize the customized part of the scheduler
        self.sche_init(**sche_conf)


    @abstractmethod
    def sche_init(self, **sche_conf) -> List[str]:
        """
        The initialization function where the scheduler part of the optimscheduler is initialized.
        Mainly decide how the learning rate adjustment strategy works as the training goes.

        Args:
            **sche_conf: Dict
                The customized arguments used to initialize the scheduler part of this optimeduler.

        """
        raise NotImplementedError


    def step(self, losses: Dict[str, torch.Tensor], scaler: GradScaler,
             time_func, optim_name: str, step_num: int):
        """
        The function that updates the target parameters of the model with the input losses.

        Args:
            losses: Dict
                The input losses received from the model.
            step_num: int
                The number of the current step number. Received from the runner.

        """
        # --- loss backward part --- #
        optim_flag = False
        with time_func(["loss_backward_time", optim_name]):
            for name, loss in losses.items():
                if self.optim_losses is None or name in self.optim_losses:
                    # narrow the loss for accumulation
                    loss /= self.accum_grad
                    # backward the loss in either amp setting or normal setting
                    scaler.scale(loss).backward() if scaler is not None else loss.backward()
                    optim_flag = True

        if not optim_flag:
            raise RuntimeError(f"No losses are back-propagated in the optimeduler {self.__class__.__name__}! "
                               f"This means that optim_losses doesn't exist in the input losses! "
                               f"Please check your input names in optim_losses.")

        # --- model optimization part --- #
        with time_func(["optim_time", optim_name]):
            real_step = (step_num - 1) // self.accum_grad + 1
            # do something only when the real step number meets the updating interval
            if real_step % self.step_per_update == 0:
                # update the learning rate for the current step (scaled by the finetuning factor)
                curr_lr = self.update_lr(real_step=real_step)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.ft_factor * curr_lr

                # update the model parameters if the accumulation interval is met
                if step_num % self.accum_grad == 0:
                    # load the stored gradients to the model if we have
                    if self.tmp_grads is not None:
                        for name, params in self.model.named_parameters():
                            params.grad += self.tmp_grads[name].cuda(device=params.grad.device)

                        # reset self.tmp_grads after loading it into the model
                        self.tmp_grads = None

                    # unscale the gradients in advance to enable clipping in the amp setting
                    # refer: https://pytorch.org/docs/1.10/amp.html#torch.cuda.amp.GradScaler.unscale_
                    if scaler is not None:
                        scaler.unscale_(self.optimizer)

                    # apply the gradient clipping right before updating the target parameters
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.grad_clip,
                        norm_type=self.grad_norm_type,
                    )

                    # optimize the target parameters only when the values of gradients are finite
                    if not torch.isfinite(grad_norm):
                        warnings.warn(f"The grad_norm in the current step is {grad_norm}! "
                                      f"The parameters are not updated in this step.")
                        if scaler is not None:
                            scaler.step(self.optimizer)
                            scaler.update()
                    else:
                        if scaler is not None:
                            scaler.step(self.optimizer)
                            scaler.update()
                        else:
                            self.optimizer.step()

                # otherwise, store the calculated gradients of the target model parameters
                else:
                    _grads = OrderedDict()
                    for name, params in self.model.named_parameters():
                        _grad = params.grad
                        if _grad is not None:
                            # transform the gradients to cpu for GPU memory saving
                            _grads[name] = _grad.detach().cpu()
                        else:
                            raise RuntimeError(f"Parameters {name} doesn't receive gradients during training!"
                                               f"Got {name}.grad = {_grad}!")

                    # store the gradients into self.tmp_grads
                    if self.tmp_grads is None:
                        self.tmp_grads = _grads
                    else:
                        # get the keys that exist in self.tmp_grads but doesn't exist in _grads
                        exclusive_keys = set(self.tmp_grads.keys()).difference(set(_grads.keys()))

                        # sum up stored gradients self.tmp_grads and new gradients _grads if the keys are not exclusive
                        if len(exclusive_keys) == 0:
                            for name in self.tmp_grads.keys():
                                self.tmp_grads[name] += _grads[name]
                        else:
                            raise RuntimeError(f"There are exclusive keys in self.tmp_grads compared to _grads. "
                                               f"The keys {exclusive_keys} don't exist in _grads!")

        # flush the gradients of the entire model to 0 no matter whether the gradients are used or stored
        self.model.zero_grad()


    @abstractmethod
    def update_lr(self, real_step: int):
        """
        The function where the learning rate is adjusted according to the input step number.

        Note that the input step number must the one of the real step. The real step number means the times of
        updating parameters. For example, if accum_grad > 1, the step_num received from the runner is not the real
        step number.

        Args:
            real_step: int
                The real step number

        """
        raise NotImplementedError


    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


    def state_dict(self):
        return dict(
            optimizer=self.optimizer.state_dict(),
            tmp_grads=self.tmp_grads
        )


    def load_state_dict(self, state_dict: Dict[str, Any]):
        # load the optimizer
        self.optimizer.load_state_dict(state_dict['optimizer'])
        # load the temporary gradients
        self.tmp_grads = state_dict['tmp_grads']


    @abstractmethod
    def __repr__(self):
        raise NotImplementedError
