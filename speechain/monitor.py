"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import copy
import os
import argparse
import time
import GPUtil
from contextlib import contextmanager

import torch
import numpy as np

from typing import Dict, List
from abc import ABC, abstractmethod
from multiprocessing import Process, Queue, Event

from speechain.model.abs import Model
from speechain.snapshooter import snapshot_logs



class Monitor(ABC):
    """
    The base class for all the monitors in this toolkit.

    """
    def __init__(self, logger, args: argparse.Namespace, result_path: str = None, **kwargs):
        """

        Args:
            logger:
            args:
            **kwargs:
        """
        # shared members of all the monitors
        self.logger = logger
        self.result_path = args.result_path if result_path is None else result_path
        self.gpus = args.gpus if isinstance(args.gpus, List) else [args.gpus]

        # shared record information for all monitors
        # epoch-level records
        self.epoch_records = dict(
            consumed_time=dict(
                data_load_time=[],
                model_forward_time=[]
            ),
            consumed_memory=dict(),
            criteria=dict()
        )
        for rank, gpu in enumerate(self.gpus):
            if f'Rank{rank}' not in self.epoch_records['consumed_memory'].keys():
                self.epoch_records['consumed_memory'][f'Rank{rank}'] = []
        # step-level records
        self.step_records = dict(
            consumed_time=dict(
                data_load_time=[],
                model_forward_time=[]
            ),
            criteria=dict()
        )
        self.mode = None
        self.monitor_init(args, **kwargs)

        # initialize the snapshooter of the monitor
        self.logs_queue = Queue()
        snapshot_conf = dict(
            result_path=self.result_path, snap_mode=self.mode, **args.monitor_snapshot_conf
        )
        # initialize the multiprocessing event to enable the communication with snapshooter process
        self.event = Event()
        self.event.clear()
        Process(target=snapshot_logs, args=(self.logs_queue, self.event, snapshot_conf), daemon=True).start()


    def enqueue(self, logs: Dict or List[Dict]):
        """

        Args:
            logs:

        Returns:

        """
        if isinstance(logs, Dict):
            self.logs_queue.put(logs)
        elif isinstance(logs, List):
            for log in logs:
                self.logs_queue.put(log)
        else:
            raise RuntimeError


    def empty_queue(self):
        return self.logs_queue.empty()


    @contextmanager
    def measure_time(self, names: str or List[str]):
        """

        Args:
            names:

        Returns:

        """
        start = time.perf_counter()
        yield
        t = time.perf_counter() - start

        names = ["consumed_time", names] if isinstance(names, str) else ["consumed_time"] + names
        dict_pointer = self.step_records
        for i, name in enumerate(names):
            if name not in dict_pointer.keys():
                dict_pointer[name] = [] if i == len(names) - 1 else dict()
            dict_pointer = dict_pointer[name]
        dict_pointer.append(t)


    def refresh_step_records(self, records: Dict = None):
        """

        Args:
            records:

        Returns:

        """
        if records is None:
            records = self.step_records
        if isinstance(records, Dict):
            for key in records.keys():
                if isinstance(records[key], Dict):
                    self.refresh_step_records(records[key])
                elif isinstance(records[key], List):
                    records[key] = []
                else:
                    raise RuntimeError
        else:
            raise RuntimeError


    def record_step_info(self, key: str, step_info: Dict):
        """

        Args:
            key:
            step_info:

        Returns:

        """
        for name, info in step_info.items():
            if name not in self.step_records[key].keys():
                self.step_records[key][name] = []
            # result is in the form of torch.Tensor, so it needs to be transformed by .item()
            if isinstance(info, (torch.Tensor, np.ndarray)):
                if len(info.shape) == 1:
                    info = info[0]
                info = info.item()
            self.step_records[key][name].append(info)


    def record_consumed_time(self, epoch_message: str):
        """

        Args:
            epoch_message:

        Returns:

        """
        epoch_message += " -- Consumed Time -- \n"
        # record the data loading time
        _total_time = sum(self.step_records['consumed_time']['data_load_time'])
        epoch_message += f"Total data load time: {_total_time:.2f}s -- "
        self.epoch_records['consumed_time']['data_load_time'].append(_total_time)

        # record the model forward time
        _total_time = sum(self.step_records['consumed_time']['model_forward_time'])
        epoch_message += f"Total model forward time: {_total_time:.2f}s -- "
        self.epoch_records['consumed_time']['model_forward_time'].append(_total_time)
        epoch_message += "\n"

        return epoch_message


    def record_consumed_memory(self, epoch_message: str):
        """

        Args:
            epoch_message:

        Returns:

        """
        epoch_message += " -- Consumed Memory -- \n"
        gpus = GPUtil.getGPUs()
        if len(gpus) == 0:
            self.logger.warn(f"GPUtil.getGPUs() returns nothing at the {self.mode} part of epoch no.{self.epoch}. ")

        for rank, gpu in enumerate(self.gpus):
            # --- torch.cuda is only able to report the GPU used in the current rank --- #
            # --- but torch.cuda can report the precise allocated and reserved memory information of the model --- #
            # turn bytes into MB，
            # memory_allocated = int(torch.cuda.memory_allocated(gpu) * (10 ** (-6)))
            # memory_reserved = int(torch.cuda.memory_reserved(gpu) * (10 ** (-6)))
            # epoch_message += f"GPU rank no.{rank} (cuda:{gpu}): " \
            #                  f"allocated memory {memory_allocated} MB, " \
            #                  f"reserved memory {memory_reserved} MB -- "
            # self.epoch_records['consumed_memory'][f'Rank{rank}']['memory_allocated'].append(memory_allocated)
            # self.epoch_records['consumed_memory'][f'Rank{rank}']['memory_reserved'].append(memory_reserved)

            # --- GPUtil can load the information of all the GPUs no matter the rank of the current process --- #
            # --- but it sometimes fails to report anything because of some IO errors --- #
            # --- and the returned used memory is the overall memory consumption of the GPU, --- #
            # --- which means that if there are more than one jobs running on the same GPU, --- #
            # --- the used memory is not precise for the current job. --- #
            memory_used = 0
            if len(gpus) != 0:
                memory_used = gpus[gpu].memoryUsed
            epoch_message += f"GPU rank no.{rank} (cuda:{gpu}): {memory_used} MB -- "
            self.epoch_records['consumed_memory'][f'Rank{rank}'].append(memory_used)
        epoch_message += "\n"

        return epoch_message


    def record_criteria(self, epoch_message: str):
        """

        Args:
            epoch_message:

        Returns:

        """
        epoch_message += " -- Criteria information -- \n"
        # loop all the training criteria
        for name, results in self.step_records['criteria'].items():
            if name not in self.epoch_records['criteria'].keys():
                self.epoch_records['criteria'][name] = []

            # calculate the average criterion value
            aver_result = np.mean(results).item()
            std_result = np.std(results).item()
            epoch_message += f"Average {name}: {aver_result:.2f} ± {std_result:.2f}\n"
            # record the average criterion value
            self.epoch_records['criteria'][name].append(aver_result)
        epoch_message += "\n"

        return epoch_message


    @abstractmethod
    def monitor_init(self, args: argparse.Namespace, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def start_epoch(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def step(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def finish_epoch(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def state_dict(self):
        raise NotImplementedError

    def load_state_dict(self, state_dict: Dict):
        for key, value in state_dict:
            self.__setattr__(key, value)


class TrainMonitor(Monitor):
    """
    The object used to monitor the training process and give the real-time logging information.

    """
    def monitor_init(self, args: argparse.Namespace):
        # general members
        self.report_per_steps = args.report_per_steps
        self.dry_run = args.dry_run
        self.no_optim = args.no_optim
        self.mode = 'train'

        # training monitor needs to additionally track optimizer information
        # update epoch-level records
        self.epoch_records['consumed_time'].update(
            loss_backward_time=dict(),
            optim_time=dict()
        )
        self.epoch_records.update(
            optim_lr=dict()
        )
        # update step-level records
        self.step_records['consumed_time'].update(
            loss_backward_time=dict(),
            optim_time=dict()
        )
        self.step_records.update(
            optim_lr=dict()
        )


    def start_epoch(self, epoch: int):
        """
        Initialize the monitor information.

        Args:
            epoch: int
                The number of the current epoch.

        Returns:
            The logging information of starting the given part of the current epoch.

        """
        # epoch-level information
        self.epoch = epoch
        self.epoch_start_time = time.time()

        # refresh the step-level records at the beginning of each epoch
        self.refresh_step_records()

        # logging the beginning information
        self.logger.info(f"The training part of epoch no.{epoch} starts.")


    def step(self, step_num: int, optim_lr: Dict[str, float], train_metrics: Dict[str, torch.Tensor]):
        """
        Record and report the information in each step.

        Args:
            step_num: int
                The number of the current training step
            optim_lr: List
                The information of each OptimScheduler in this step. Including optimization time and learning rates.
            train_metrics: Dict
                The criterion results of the model forward.

        Returns:
            The step message for the logger to log. None means nothing to log.

        """
        # accumulate the values of training criteria
        if train_metrics is not None:
            self.record_step_info('criteria', train_metrics)

        # accumulate the optimization times and learning rates of each OptimScheduler
        if optim_lr is not None:
            self.record_step_info('optim_lr', optim_lr)

        # report all the information for every 'report_per_steps' training steps
        if step_num % self.report_per_steps == 0:
            # calculate the accumulated time for reporting
            _data_load_time = sum(self.step_records['consumed_time']['data_load_time'][-self.report_per_steps:])
            _model_forward_time = sum(self.step_records['consumed_time']['model_forward_time'][-self.report_per_steps:])
            # initialize the returned message of the current step
            step_message = f"Training step no.{step_num - self.report_per_steps + 1:d}-{step_num:d} -- " \
                           f"data loading time: {_data_load_time:.2f}s -- " \
                           f"model forward time: {_model_forward_time:.2f}s -- "

            if not self.dry_run:
                # report the values of criteria in each training step
                step_message += "Training Criteria: "
                for name, result in self.step_records['criteria'].items():
                    _tmp_criteria = result[-self.report_per_steps:]
                    step_message += f"{name}: {np.mean(_tmp_criteria):.2f} ± {np.std(_tmp_criteria):.2f} -- "

                if not self.no_optim:
                    # report the information of optimizers after each training step
                    step_message += "OptimSchedulers: "
                    for optim_name in self.step_records['optim_lr'].keys():
                        # accumulate the backward and optimization times
                        _loss_backward_time = sum(
                            self.step_records['consumed_time']['loss_backward_time'][optim_name][-self.report_per_steps:])
                        _optim_time = sum(
                            self.step_records['consumed_time']['optim_time'][optim_name][-self.report_per_steps:])
                        # average the learning rate
                        _lr = sum(self.step_records['optim_lr'][optim_name][-self.report_per_steps:]) / self.report_per_steps

                        # accumulated optimization time and averaged learning_rates are reported
                        step_message += f"{optim_name}: " \
                                        f"loss backward time= {_loss_backward_time:.2f}s, " \
                                        f"optimization time={_optim_time:.2f}s, " \
                                        f"learning rate={_lr:.2e} -- "

            # logging the information of the current step
            self.logger.info(step_message)


    def finish_epoch(self):
        """
        Logging the epoch information and making the snapshots for the passed epoch

        """
        # ---- The Information Logging Part ---- #
        # report the overall consuming time of the current epoch
        epoch_message = f"The training part of epoch no.{self.epoch} is finished in {time.time() - self.epoch_start_time:.2f}s.\n" \
                        f"Summary of all training steps:\n"

        # report the information of the consumed calculation time
        epoch_message = self.record_consumed_time(epoch_message)

        # report the information of the consumed GPU memory
        epoch_message = self.record_consumed_memory(epoch_message)

        # data loading only
        if not self.dry_run:
            # report the information of all the training criteria
            epoch_message = self.record_criteria(epoch_message)

            # no optimization
            if not self.no_optim:
                # report the information of all the OptimSchedulers
                epoch_message += " -- OptimScheduler information -- \n"
                # record the optimization information of the current epoch
                for optim_name in self.step_records['optim_lr'].keys():
                    if optim_name not in self.epoch_records['optim_lr'].keys():
                        self.epoch_records['optim_lr'][optim_name] = []
                    if optim_name not in self.epoch_records['consumed_time']['loss_backward_time'].keys():
                        self.epoch_records['consumed_time']['loss_backward_time'][optim_name] = []
                    if optim_name not in self.epoch_records['consumed_time']['optim_time'].keys():
                        self.epoch_records['consumed_time']['optim_time'][optim_name] = []

                    epoch_message += f"{optim_name} -- "
                    # accumulate the loss backward time
                    _total_time = sum(self.step_records['consumed_time']['loss_backward_time'][optim_name])
                    self.epoch_records['consumed_time']['loss_backward_time'][optim_name].append(_total_time)
                    epoch_message += f"Total loss backward time: {_total_time:.2f}s, "

                    # accumulate the optimization time
                    _total_time = sum(self.step_records['consumed_time']['optim_time'][optim_name])
                    self.epoch_records['consumed_time']['optim_time'][optim_name].append(_total_time)
                    epoch_message += f"Total optimization time: {_total_time:.2f}s, "

                    # average the learning rate
                    aver_lr = np.mean(self.step_records['optim_lr'][optim_name]).item()
                    self.epoch_records['optim_lr'][optim_name].append(aver_lr)
                    epoch_message += f"Average learning rate: {aver_lr:.2e}\n"
                epoch_message += "\n"

        # logging the information for the current epoch
        self.logger.info(epoch_message)


        # ---- The SnapShotting Part ---- #
        for key in self.epoch_records.keys():
            # only snapshot the time records in the dry running mode
            if self.dry_run and key != 'consumed_time':
                continue
            # skip the learning rate records in the no optimization mode
            if self.no_optim and key == 'optim_lr':
                continue
            # snapshot the epoch records so for to a curve figure
            self.enqueue(
                dict(
                    materials=copy.deepcopy(self.epoch_records[key]), plot_type='curve',
                    epoch=self.epoch, xlabel="epoch", sep_save=False, subfolder_names=key
                )
            )

        # notify the snapshooter process of the new queue elements
        self.event.set()


    def state_dict(self):
        """
        Save the information of all the recorded epochs

        """
        return dict(
            epoch_records=self.epoch_records
        )


class ValidMonitor(Monitor):
    """
    The object used to monitor the validation process and give the real-time logging information.

    """
    def monitor_init(self, args: argparse.Namespace, model: Model):
        # register a pointer of the model
        self.model = model

        # running mode
        self.dry_run = args.dry_run
        self.no_optim = args.no_optim
        self.mode = 'valid'

        # best models-related members
        assert args.best_model_mode.lower() in ['max', 'min'], \
            f"The best_model_mode you give to the monitor must be either 'max' or 'min', but got {args.best_model_mode}."
        self.best_model_num = args.best_model_num
        self.best_model_mode = args.best_model_mode.lower()
        self.best_model_metric = args.best_model_metric
        self.best_model_performance = dict()

        self.model_save_path = os.path.join(self.result_path, 'models')
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path, exist_ok=True)

        # early stopping-related members
        self.early_stopping_patience = args.early_stopping_patience
        self.early_stopping_threshold = args.early_stopping_threshold
        self.early_stopping_epochs = 0

        # initialize the snapshooter of this validation monitor
        self.model_snapshot_interval = args.model_snapshot_interval


    def start_epoch(self, epoch: int):
        """
        Initialize the monitor information.

        Args:
            epoch: int
                The number of the current epoch.

        """
        # epoch-level information
        if epoch in self.best_model_performance.keys():
            self.logger.warning(f"The record of epoch no.{epoch} has already existed in the monitor! "
                                f"It will be overwritten by the new record obtained shortly thereafter.")
        self.epoch = epoch
        self.epoch_start_time = time.time()

        # refresh the step-level records at the beginning of each epoch
        self.refresh_step_records()

        # logging the beginning information
        self.logger.info(f"The validation part of epoch no.{epoch} starts.")


    def step(self, valid_metrics: Dict[str, torch.Tensor]):
        """
        Record and report the information in each validation step.

        Args:
            valid_metrics: Dict
                The validation criterion results of the model forward.

        """
        # accumulate the values of validation criteria
        if valid_metrics is not None:
            self.record_step_info('criteria', valid_metrics)


    def model_snapshot(self, epoch: int, sample_index: str, used_sample: Dict):
        """

        Args:
            epoch:
            sample_index:
            used_sample:

        Returns:

        """
        # initialize the sub-dict for each sample
        if sample_index not in self.epoch_records.keys():
            self.epoch_records[sample_index] = dict()

        # get the visualization logs for model snapshotting
        vis_logs = self.model(batch_data=used_sample, epoch_records=self.epoch_records,
                              epoch=epoch, sample_index=sample_index,
                              snapshot_interval=self.model_snapshot_interval)

        # put all the visualization logs into the queue
        self.enqueue(vis_logs)


    def is_better(self, query, target):
        """

        Args:
            query:
            target:

        Returns:

        """
        _target = target
        # relative threshold
        if self.early_stopping_threshold > 0:
            _target *= 1 + self.early_stopping_threshold if self.best_model_mode == 'max' \
                else 1 - self.early_stopping_threshold
        # absolute threshold
        elif self.early_stopping_threshold < 0:
            _target += -self.early_stopping_threshold if self.best_model_mode == 'max' \
                else self.early_stopping_threshold

        # the threshold is applied to the better comparison
        return query > _target if self.best_model_mode.lower() == 'max' else query < _target


    def model_insert(self, curr_performance: int or float):
        """
        Insert the model in the current epoch into self.best_model_performance

        Args:
            curr_performance:

        Returns:

        """
        # controls whether to insert the current model into self.best_model_performance or not
        model_insert_flag = False

        # if no empty positions for the best models
        if len(self.best_model_performance) == self.best_model_num:
            # as long as the current performance is better than one existing record, it should be inserted.
            for performance in self.best_model_performance.values():
                if self.is_better(curr_performance, performance):
                    model_insert_flag = True
                    break
        # True if there are some empty positions for the best models
        else:
            model_insert_flag = True

        if model_insert_flag:
            # save the model of the current epoch to the disk
            torch.save(self.model.state_dict(), os.path.join(self.model_save_path, f"epoch_{self.epoch}.mdl"))
            # record the performance of the current epoch
            self.best_model_performance[self.epoch] = curr_performance


    def save_best_and_pop_worst(self, epoch_message: str):
        """

        Args:
            epoch_message:

        Returns:

        """
        # find the best epoch and worst epoch in self.best_model_performance
        sorted_epochs = dict(sorted(self.best_model_performance.items(), key=lambda x: x[1],
                                    reverse=True if self.best_model_mode == 'max' else False))
        sorted_epochs = list(sorted_epochs.keys())
        best_epoch, worst_epoch = sorted_epochs[0], sorted_epochs[-1]

        # controls whether the worst model has been poped out or not
        worst_pop_flag = False

        # pop out the worst model if there is a redundant one in self.best_model_performance
        if len(self.best_model_performance) > self.best_model_num:
            self.best_model_performance.pop(worst_epoch)
            sorted_epochs.remove(worst_epoch)
            os.remove(os.path.join(self.model_save_path, f"epoch_{worst_epoch}.mdl"))
            worst_pop_flag = True

        # update all the best models so far
        for i, epoch in enumerate(sorted_epochs):
            _best_model_pointer = f"{self.best_model_metric}_best.mdl" if i == 0 else \
                f"{self.best_model_metric}_best_{i + 1}.mdl"
            # create a soft link from the best model pointer to the model file of the current epoch
            symlink_dst = os.path.join(self.model_save_path, _best_model_pointer)
            if os.path.islink(symlink_dst) or os.path.exists(symlink_dst):
                os.unlink(symlink_dst)
            os.symlink(os.path.join(self.model_save_path, f"epoch_{epoch}.mdl"), symlink_dst)

        # whether the current model is the best one so far
        best_save_flag = False
        if best_epoch == self.epoch:
            best_save_flag = True
            epoch_message += f"{self.best_model_metric} of epoch no.{self.epoch} is the best so far. " \
                             f"The parameters of the model in epoch no.{self.epoch} has been stored.\n"

        return epoch_message, best_save_flag, worst_pop_flag


    def save_aver_model(self, epoch_message: str, worst_pop_flag: bool):
        """
        save the average models of the best models

        Args:
            epoch_message:
            worst_pop_flag:

        Returns:

        """
        # average the recorded best models
        if len(self.best_model_performance) == self.best_model_num and worst_pop_flag:
            # sum up the parameters of all best models
            avg_model = None
            for epoch in self.best_model_performance.keys():
                _avg = None
                if avg_model is not None:
                    _avg = torch.load(os.path.join(self.model_save_path, f"epoch_{epoch}.mdl"), map_location="cpu")
                else:
                    avg_model = torch.load(os.path.join(self.model_save_path, f"epoch_{epoch}.mdl"), map_location="cpu")

                if _avg is not None:
                    for key in avg_model.keys():
                        avg_model[key] += _avg[key]

            # for the parameters whose dtype is int, averaging is not performed
            # reference: https://github.com/espnet/espnet/blob/5fa6dcc4e649dc66397c629d0030d09ecef36b80/espnet2/main_funcs/average_nbest_models.py#L90
            for key in avg_model.keys():
                if not str(avg_model[key].dtype).startswith("torch.int"):
                    avg_model[key] /= len(self.best_model_performance)

            # save the average model
            _aver_model_path = os.path.join(self.model_save_path,
                                            f"{self.best_model_num}_{self.best_model_metric}_average.mdl")
            torch.save(avg_model, _aver_model_path)

            # report to the logger
            epoch_message += f"Best {self.best_model_num:d} models so far has been updated to {list(self.best_model_performance.keys())}. " \
                             f"The average model has been stored to {_aver_model_path}."

        return epoch_message


    def check_early_stop(self, epoch_message: str, best_save_flag: bool):
        """
        Check whether to stop the training process early or not

        Args:
            epoch_message:
            best_save_flag:

        Returns:

        """

        # update the number of early stopping epoch if the current model is not the best one
        early_stopping_flag = False
        if not best_save_flag:
            self.early_stopping_epochs += 1
            epoch_message += f"\nNo improvement of {self.best_model_metric} in epoch no.{self.epoch}. " \
                             f"The best.mdl has already not been updated for {self.early_stopping_epochs} epochs.\n"

            if self.early_stopping_epochs > self.early_stopping_patience:
                early_stopping_flag = True
                epoch_message += f"The early_stopping_patience {self.early_stopping_patience} is reached, " \
                                 f"so the training process stops here."
        else:
            self.early_stopping_epochs = 0

        return epoch_message, early_stopping_flag


    def finish_epoch(self):
        """
        This function contains the logic of early stopping, best models updating, models averaging.

        """
        # ---- The Information Logging Part ---- #
        # report the overall consuming time of the current validation epoch
        epoch_message = f"The validation part of epoch no.{self.epoch} is finished in {time.time() - self.epoch_start_time:.2f}s.\n" \
                        f"Summary of all validation steps:\n"

        # report the information of the consumed calculation time
        epoch_message = self.record_consumed_time(epoch_message)

        # report the information of the consumed GPU memory
        epoch_message = self.record_consumed_memory(epoch_message)

        if not self.dry_run:
            # report the information of all the validation criteria
            epoch_message = self.record_criteria(epoch_message)

        # ---- The SnapShotting Part ---- #
        for key in self.epoch_records.keys():
            # only snapshot the time and memory info in the dry running mode
            if self.dry_run and key != 'consumed_time':
                continue
            # skip the model visualization records
            elif key in ['consumed_time', 'consumed_memory', 'criteria']:
                # snapshot the epoch records so for to a curve figure
                self.enqueue(
                    dict(
                        materials=copy.deepcopy(self.epoch_records[key]), plot_type='curve',
                        epoch=self.epoch, xlabel="epoch", sep_save=False, subfolder_names=key
                    )
                )

        # notify the snapshooter process of the new queue elements
        self.event.set()

        # ---- The Model Saving and Early Stopping Part ---- #
        early_stopping_flag = False
        if not self.dry_run:
            assert self.best_model_metric in self.step_records['criteria'].keys(), \
                f"The best_model_metric {self.best_model_metric} has not been calculated during the validation."

            # insert the current model into self.best_model_performance if needed
            self.model_insert(self.epoch_records['criteria'][self.best_model_metric][-1])

            # After inserting, find the best epoch and worst epoch in self.best_model_performance
            epoch_message, best_save_flag, worst_pop_flag = self.save_best_and_pop_worst(epoch_message)

            # save the average models of the best models so far if needed
            epoch_message = self.save_aver_model(epoch_message, worst_pop_flag)

            # controls whether to stop the training process early or not
            epoch_message, early_stopping_flag = self.check_early_stop(epoch_message, best_save_flag)

        # log the information of the current validation epoch
        self.logger.info(epoch_message)
        return early_stopping_flag


    def state_dict(self):
        """
        Save the best models and current number of early-stopping epochs

        Returns:

        """
        return dict(
            epoch_records=self.epoch_records,
            best_model_performance=self.best_model_performance,
            early_stopping_epochs=self.early_stopping_epochs
        )



class TestMonitor(Monitor):
    """
    The object used to monitor the training process and give the real-time logging information.

    """

    def monitor_init(self, args: argparse.Namespace, model: Model):
        """

        Args:
            args:
            model:

        Returns:

        """
        self.model = model
        self.report_per_steps = args.report_per_steps

        # aver_required_metrics in the input configuration is given the highest priority
        if 'aver_required_metrics' in args and args.aver_required_metrics is not None:
            if not isinstance(args.aver_required_metrics, List):
                args.aver_required_metrics = [args.aver_required_metrics]
            self.aver_required_metrics = args.aver_required_metrics

        # if aver_required_metrics is not given in the input configuration, use the default value of the model
        elif hasattr(self.model, 'aver_required_metrics'):
            if not isinstance(self.model.aver_required_metrics, List):
                self.model.aver_required_metrics = [self.model.aver_required_metrics]
            self.aver_required_metrics = self.model.aver_required_metrics

        # otherwise, no metrics will be averaged
        else:
            self.aver_required_metrics = None


    def start_epoch(self, total_step_num: int):
        """
        For the evaluation stage, we only need to initialize the step_info to register the results

        """
        # para init
        test_message = "The evaluation stage starts.\n"
        self.prev_test_time = time.time()
        self.total_step_num = total_step_num
        self.step_info = dict(
            group_time=[],
            total_time=0
        )
        self.group_num = 1

        # initialize the metrics required to calculate their average values
        if self.aver_required_metrics is not None:
            test_message += f"The testing metrics {self.aver_required_metrics} will be averaged."
        else:
            test_message += f"No testing metrics will be averaged."
        self.logger.info(test_message)


    def step(self, step_num: int, test_results: Dict, test_index: str):
        """

        Args:
            test_results:
            args:
            test_index:
            subset:

        Returns:

        """

        test_step_message = None
        # register the metric results in the testing stage
        for name, result in test_results.items():
            _result_path = os.path.join(self.result_path, name)
            if step_num == 1 and os.path.exists(_result_path):
                os.remove(_result_path)
            with open(_result_path, 'ab') as f:
                np.savetxt(f, [[test_index, result]], fmt="%s")

            if name in self.aver_required_metrics:
                if name not in self.step_info.keys():
                    self.step_info[name] = []
                self.step_info[name].append(result)

        if step_num % self.report_per_steps != 0:
            _curr_test_time = time.time()
            self.step_info['group_time'].append(_curr_test_time - self.prev_test_time)
            self.prev_test_time = _curr_test_time

        else:
            _group_test_time = sum(self.step_info['group_time'])
            self.step_info['total_time'] += _group_test_time
            self.step_info['group_time'] = []

            _finish_step_num = int(self.group_num * self.report_per_steps)
            _remaining_step_num = self.total_step_num - _finish_step_num
            _remaining_time = (self.step_info['total_time'] / self.group_num) * (_remaining_step_num / self.report_per_steps)
            self.group_num += 1

            test_step_message = f"Testing Midway Report -- " \
                                f"testing time for the recent {self.report_per_steps} steps: {_group_test_time:.2f}s -- " \
                                f"finished step number: {_finish_step_num} -- " \
                                f"remaining step number: {_remaining_step_num} -- " \
                                f"expected remaining time: "

            remaining_days, _remaining_time = int(_remaining_time // (3600 * 24)), _remaining_time % (3600 * 24)
            if remaining_days > 0:
                test_step_message += f"{remaining_days:d}d "

            remaining_hours, _remaining_time = int(_remaining_time // 3600), _remaining_time % 3600
            if remaining_hours > 0:
                test_step_message += f"{remaining_hours:d}h "

            remaining_minutes, _remaining_time = int(_remaining_time // 60), _remaining_time % 60
            if remaining_minutes > 0:
                test_step_message += f"{remaining_minutes:d}m "

            remaining_seconds = _remaining_time
            test_step_message += f"{remaining_seconds:.2f}s"

        if test_step_message is not None:
            self.logger.info(test_step_message)


    def finish_epoch(self):
        """

        Returns:

        """
        test_message = ""
        result_txt = dict()
        result_txt_path = os.path.join(self.result_path, "result")

        if self.aver_required_metrics is not None:
            for name, results in self.step_info.items():
                if name in self.aver_required_metrics:
                    result_txt[name] = f"{np.mean(results):2f} ± {np.std(results):2f}"

            np.savetxt(result_txt_path, [[key, value] for key, value in result_txt.items()], fmt="%s")
            test_message += f"{self.aver_required_metrics} metrics has been saved into {result_txt_path}."

        self.logger.info(test_message)


    def state_dict(self):
        """
        Save the best models and current number of early-stopping epochs

        Returns:

        """
        return