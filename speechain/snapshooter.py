"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""

import math
import os
import warnings
from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

try:
    import soundfile as sf
except OSError:
    pass
from abc import ABC, abstractmethod
from multiprocessing import Event, Queue
from typing import Any, Dict, List

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter


class Plotter(ABC):
    """ Plotter is the base class for all plotters.
    """

    @abstractmethod
    def __init__(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def plot(
        self, ax, material: Any, fig_name: str, xlabel: str, ylabel: str, **kwargs
    ):
        raise NotImplementedError

    @abstractmethod
    def save(self, save_path: str, **kwargs):
        raise NotImplementedError


class CurvePlotter(Plotter):
    """ CurvePlotter does the job of plotting the curve of the given material. 
    """

    def __init__(self, plot_conf: Dict = None, grid_conf: Dict = None):
        """ Initialize the CurvePlotter.

        Args:
            plot_conf (Dict): the configuration of the plot
            grid_conf (Dict): the configuration of the grid
        """
        # default arguments of the plot_conf
        self.plot_conf = dict(linestyle="-", linewidth=1, marker="o", markersize=5)
        # overwrite the plot_conf by the input arguments
        if plot_conf is not None:
            for key, value in plot_conf.items():
                self.plot_conf[key] = value

        # default arguments of the plot_conf
        self.grid_conf = dict(linestyle="--", linewidth=1, color="black", alpha=0.3)
        # overwrite the plot_conf by the input arguments
        if grid_conf is not None:
            for key, value in grid_conf.items():
                self.grid_conf[key] = value

    def plot(
        self,
        ax,
        material: List or Dict[str, List],
        fig_name: str,
        xlabel: str,
        ylabel: str,
        x_stride: int = 1,
    ):
        """ Plot the curve of the given material.

        Args:
            ax (matplotlib.axes._subplots.AxesSubplot): the axis of the figure
            material (List or Dict[str, List]): the material to be plotted
            fig_name (str): the name of the figure
            xlabel (str): the label of x-axis
            ylabel (str): the label of y-axis
            x_stride (int): the stride of the x-axis

        Returns:
            None    
        """
        # set up the figure label of x and y axes
        ax.set_xlabel(xlabel if xlabel is not None else None)
        ax.set_ylabel(ylabel if ylabel is not None else fig_name)

        # set the figure grid
        ax.grid(**self.grid_conf)

        # only one curve in the figure
        if isinstance(material, List):
            # for the stand-alone figure, only show up to 5 points at x-axis
            x_axis = np.arange(0, len(material), dtype=np.int_) * x_stride + 1
            interval = math.ceil(len(x_axis) / 5)
            ax.set_xticks(
                x_axis,
                [
                    str(x_axis[i]) if i % interval == 0 else ""
                    for i in range(len(x_axis))
                ],
            )
            ax.plot(x_axis, material, **self.plot_conf)

        # multiple curves in the figure
        elif isinstance(material, Dict):
            if len(material) > 0:
                keys = list(material.keys())
                # for the summary figure, only show up to 5 points at x-axis
                x_axis = (
                    np.arange(1, len(material[keys[0]]) + 1, dtype=np.int_) * x_stride
                )
                interval = math.ceil(len(x_axis) / 5)
                ax.set_xticks(
                    x_axis,
                    [
                        str(x_axis[i]) if i % interval == 0 else ""
                        for i in range(len(x_axis))
                    ],
                )
                for key in keys:
                    ax.plot(x_axis, material[key], label=key, **self.plot_conf)
                plt.legend()

    def save(self, materials: Dict, save_path: str, x_stride: int = 1):
        """ Save the given material into a specific .txt file for easy visualization.

        Args:
            materials (Dict): the material to be saved
            save_path (str): the path to save the materials
            x_stride (int): the stride of the x-axis

        Returns:

        """
        # save each material into a specific .txt file for easy visualization
        for file_name, material in materials.items():
            # only one file
            if isinstance(material, List):
                # initialize the horizontal axis
                x_axis = np.arange(1, len(material) + 1, dtype=np.int_) * x_stride
                save_data = np.concatenate(
                    (x_axis.reshape(-1, 1), np.array(material).reshape(-1, 1)), axis=-1
                )
                np.savetxt(f"{os.path.join(save_path, file_name)}.txt", save_data)
            # multiple files in the path
            elif isinstance(material, Dict):
                if len(material) > 0:
                    keys = list(material.keys())
                    x_axis = (
                        np.arange(1, len(material[keys[0]]) + 1, dtype=np.int_)
                        * x_stride
                    )
                    for key in keys:
                        save_data = np.concatenate(
                            (
                                x_axis.reshape(-1, 1),
                                np.array(material[key]).reshape(-1, 1),
                            ),
                            axis=-1,
                        )
                        np.savetxt(
                            f"{os.path.join(save_path, '_'.join([file_name, key]))}.txt",
                            save_data,
                        )


class MatrixPlotter(Plotter):
    """ Plot the matrix of the given material.
    """

    def __init__(self, **plot_conf):
        """ Initialize the plotter.

        Args:
            **plot_conf:

        Returns:
            None
        """
        # default arguments of the plot_conf
        self.plot_conf = dict(
            cmap="Blues_r",
            cbar=False,
        )
        # overwrite the plot_conf by the input arguments
        if plot_conf is not None:
            for key, value in plot_conf.items():
                self.plot_conf[key] = value

        self.plot_conf["cmap"] = plt.get_cmap(self.plot_conf["cmap"])

    def plot(
        self,
        ax,
        material: np.ndarray,
        fig_name: str,
        xlabel: str,
        ylabel: str,
        flip_y: bool = False,
        **kwargs,
    ):

        # set the figure title
        if ylabel is not None and xlabel is not None:
            fig_title = f"{ylabel} vs. {xlabel}"
        else:
            fig_title = ylabel if ylabel is not None else fig_name
        ax.set_title(
            fig_title, fontweight="bold", color="black", verticalalignment="baseline"
        )

        # plot the curve on the figure and save the figure
        sns.heatmap(
            data=material[::-1] if flip_y else material,
            yticklabels=(
                [
                    str(i) if (i + 1) % int(material.shape[0] / 5) == 0 else ""
                    for i in range(material.shape[0] - 1, -1, -1)
                ]
                if flip_y
                else "auto"
            ),
            **self.plot_conf,
        )

    def save(self, materials: Dict[str, np.ndarray], epoch: int, save_path: str):
        """Save the materials into a .npz file on disk for future usage.

        Args:
            materials (Dict[str, np.ndarray]): the materials to be saved
            epoch (int): the epoch of the materials
            save_path (str): the path to save the materials

        Returns:
            None    
        """
        np.savez(os.path.join(save_path, f"epoch{epoch}.npz"), **materials)


class HistPlotter(Plotter):
    """HistPlotter does the job of plotting the histogram of the given material. 
    """

    def __init__(self, plot_conf: Dict = None, grid_conf: Dict = None):
        """
        Initialize the HistPlotter.

        Args:
            plot_conf (Dict): the configuration of the plot
            grid_conf (Dict): the configuration of the grid
        """
        # default arguments of the plot_conf
        self.plot_conf = dict(bins=50, edgecolor="k")
        # overwrite the plot_conf by the input arguments
        if plot_conf is not None:
            for key, value in plot_conf.items():
                self.plot_conf[key] = value

        # default arguments of the plot_conf
        self.grid_conf = dict(linestyle="--", linewidth=1, color="black", alpha=0.3)
        # overwrite the plot_conf by the input arguments
        if grid_conf is not None:
            for key, value in grid_conf.items():
                self.grid_conf[key] = value

    def plot(
        self, ax, material: List, fig_name: str, xlabel: str, ylabel: str, **kwargs
    ):
        """Plot the histogram of the given material.

        Args:
            ax:
            material:
            fig_name:
            xlabel:
            ylabel:
            **kwargs:

        Returns:

        """
        # set up the figure label of x and y axes
        ax.set_xlabel(xlabel if xlabel is not None else fig_name)
        ax.set_ylabel(ylabel if ylabel is not None else None)

        # set the figure title
        fig_title = f"{fig_name}" if fig_name is not None else f"{xlabel}"
        ax.set_title(
            fig_title, fontweight="bold", color="black", verticalalignment="baseline"
        )

        # set the figure grid
        ax.grid(**self.grid_conf)

        # plot the histogram
        ax.hist(material, **self.plot_conf)

        # calculate the median (50%), quartiles (25% & 75%) and ten-fold points (10% & 90%) of the distribution
        sorted_material, material_len = sorted(material), len(material)
        first_tenfold, first_quat, median, last_quat, last_tenfold = (
            sorted_material[int(material_len / 10)],
            sorted_material[int(material_len / 4)],
            sorted_material[int(material_len / 2)],
            sorted_material[int(material_len / 4 * 3)],
            sorted_material[int(material_len / 10 * 9)],
        )
        # plot the median as the red vertical line with '--' style
        ax.axvline(median, c="red", ls="--")
        # plot the fist and last quartiles as the red vertical line with '-.' style
        ax.axvline(first_quat, c="orange", ls="-.")
        ax.axvline(last_quat, c="orange", ls="-.")
        # plot the fist and last ten-fold points as the red vertical line with ':' style
        ax.axvline(first_tenfold, c="yellow", ls=":")
        ax.axvline(last_tenfold, c="yellow", ls=":")

    def save(self, save_path: str, **kwargs):
        pass


def snapshot_logs(
    logs_queue: Queue, event: Event, snapshooter_conf: Dict, wait_time: int = 10
):
    """Snapshot the model during training or validation.

    Args:
        logs_queue:
        event:
        snapshooter_conf:

    Returns:

    """
    snapshooter = SnapShooter(**snapshooter_conf)
    while True:
        try:
            # check whether the queue is empty every 10 seconds
            if not logs_queue.empty():
                log = logs_queue.get()
                snapshooter.snapshot(**log)
            else:
                event.wait(timeout=wait_time)
                event.clear()
        # catch all kinds of Exceptions and continue the process
        except Exception as e:
            warnings.warn(f"snapshot_logs() meets an error: {e}.")


class SnapShooter:
    """SnapShooter does the job of recording snapshots of the model during training or validation.

    """

    def __init__(
        self,
        result_path: str,
        snap_mode: str,
        fig_width: float = 6.4,
        fig_height: float = 4.8,
        curve_plotter_conf: Dict = None,
        matrix_plotter_conf: Dict = None,
        hist_plotter_conf: Dict = None,
    ):
        """ Initialize the SnapShooter.
        Args:
            result_path:
            fig_width:
            fig_height:
            curve_plotter_conf:
            matrix_plotter_conf:
            hist_plotter_conf:

        """
        # initialize the figure plotters and the tensorboard writer
        if snap_mode is not None:
            self.figure_path = os.path.join(result_path, "figures", snap_mode)
            self.tb_writer = SummaryWriter(
                log_dir=os.path.join(result_path, "tensorboard", snap_mode)
            )
        else:
            self.figure_path = os.path.join(result_path, "figures")
            self.tb_writer = None

        # initialize the default values of the figure width and height
        self.fig_width = fig_width
        self.fig_height = fig_height

        # initialize the built-in plotters
        # curve plotter
        curve_plotter_conf = (
            dict() if curve_plotter_conf is None else curve_plotter_conf
        )
        self.curve_plotter = CurvePlotter(**curve_plotter_conf)
        # matrix plotter
        matrix_plotter_conf = (
            dict() if matrix_plotter_conf is None else matrix_plotter_conf
        )
        self.matrix_plotter = MatrixPlotter(**matrix_plotter_conf)
        # histogram plotter
        hist_plotter_conf = dict() if hist_plotter_conf is None else hist_plotter_conf
        self.hist_plotter = HistPlotter(**hist_plotter_conf)

    def snapshot(
        self,
        materials: Dict,
        plot_type: str,
        epoch: int = None,
        xlabel: str = None,
        ylabel: str = None,
        sep_save: bool = True,
        sum_save: bool = True,
        tb_write: bool = True,
        data_save: bool = True,
        col_num: int = None,
        row_num: int = None,
        subfolder_names: List[str] or str = None,
        **kwargs,
    ):

        # initialize the list of sub-folder names
        subfolder_names = [] if subfolder_names is None else subfolder_names
        subfolder_names = (
            [subfolder_names] if isinstance(subfolder_names, str) else subfolder_names
        )

        # initialize the figure saving path
        if subfolder_names is not None:
            save_path = os.path.join(self.figure_path, *subfolder_names)
        else:
            save_path = self.figure_path
        os.makedirs(save_path, exist_ok=True)

        # go through different branches based on the data type
        if plot_type == "curve":
            self.curve_snapshot(
                save_path=save_path,
                subfolder_names=subfolder_names,
                materials=materials,
                epoch=epoch,
                xlabel=xlabel,
                ylabel=ylabel,
                sep_save=sep_save,
                sum_save=sum_save,
                tb_write=tb_write,
                data_save=data_save,
                col_num=col_num,
                row_num=row_num,
                **kwargs,
            )
        elif plot_type == "matrix":
            self.matrix_snapshot(
                save_path=save_path,
                subfolder_names=subfolder_names,
                materials=materials,
                epoch=epoch,
                xlabel=xlabel,
                ylabel=ylabel,
                sep_save=sep_save,
                sum_save=sum_save,
                tb_write=tb_write,
                data_save=data_save,
                col_num=col_num,
                row_num=row_num,
                **kwargs,
            )
        elif plot_type == "hist":
            self.hist_snapshot(
                save_path=save_path, materials=materials, xlabel=xlabel, ylabel=ylabel
            )
        elif plot_type == "text":
            self.text_snapshot(
                save_path=save_path,
                subfolder_names=subfolder_names,
                materials=materials,
                epoch=epoch,
                data_save=data_save,
                **kwargs,
            )
        elif plot_type == "audio":
            self.audio_snapshot(
                save_path=save_path,
                subfolder_names=subfolder_names,
                materials=materials,
                epoch=epoch,
                data_save=data_save,
                **kwargs,
            )
        else:
            raise ValueError

        if self.tb_writer is not None:
            self.tb_writer.flush()

    @contextmanager
    def sep_figure_context(self, fig_name: str, save_path: str):

        # setup the figure plotter and initialize the figure
        fig = plt.figure(figsize=[self.fig_width, self.fig_height], num=1)
        ax = fig.add_subplot(111)

        # return the necessary information for the specific plotter to plot
        yield ax

        # save the plotted figure
        plt.savefig(os.path.join(save_path, fig_name + ".png"))
        plt.close(fig=fig)

    @contextmanager
    def sum_figure_context(
        self,
        materials: Dict,
        save_path: str,
        col_num: int,
        row_num: int,
        sum_fig_name: str = "summary.png",
    ):

        # initialize the number of columns and rows in the summary figure
        if col_num is not None and row_num is not None:
            assert col_num * row_num >= len(materials)
        elif col_num is not None:
            row_num = math.ceil(len(materials) / col_num)
        elif row_num is not None:
            col_num = math.ceil(len(materials) / row_num)
        # if both col_num and row_num are not given, they should be set automatically
        else:
            # make sure that the numbers of columns and rows in the summary figure differ by no more than 1
            material_num = len(materials)
            for divisor in range(int(math.sqrt(material_num)), material_num + 1):
                quot = material_num / divisor
                if abs(divisor - quot) <= 1:
                    quot = math.ceil(quot)
                    # the larger number is assigned to the column number
                    row_num = min(divisor, quot)
                    col_num = max(divisor, quot)
                    break

        # setup the figure plotter and initialize the figure
        # scale the height and width when the number of rows or columns exceeds 3
        fig_width = self.fig_width * math.ceil(col_num / 3)
        fig_height = self.fig_height * math.ceil(row_num / 3)
        # if row_num > col_num:
        #     fig_height *= (row_num / col_num)
        # elif col_num > row_num:
        #     fig_width *= (col_num / row_num)

        fig = plt.figure(figsize=[fig_width, fig_height], num=row_num * col_num)
        material_keys = list(materials.keys())

        yield fig, material_keys, row_num, col_num

        # save the plotted figure
        fig.tight_layout()
        plt.savefig(os.path.join(save_path, sum_fig_name))
        plt.close(fig=fig)

    def curve_snapshot(
        self,
        save_path: str,
        subfolder_names: List[str],
        materials: Dict,
        epoch: int,
        xlabel: str,
        ylabel: str,
        sep_save: bool,
        sum_save: bool,
        tb_write: bool,
        data_save: bool,
        col_num: int,
        row_num: int,
        x_stride: int = 1,
    ):

        # save each material to a separate figure
        if sep_save:
            # loop each file in the given material Dict
            for fig_name in materials.keys():
                with self.sep_figure_context(fig_name, save_path) as ax:
                    # plot the current material into a figure
                    self.curve_plotter.plot(
                        ax=ax,
                        material=materials[fig_name],
                        fig_name=fig_name,
                        x_stride=x_stride,
                        xlabel=xlabel,
                        ylabel=ylabel,
                    )

        # save all the materials to a summary figure
        if sum_save:
            with self.sum_figure_context(materials, save_path, col_num, row_num) as (
                fig,
                material_keys,
                row_num,
                col_num,
            ):
                # loop each row
                for row in range(1, row_num + 1):
                    # loop each column
                    for col in range(1, col_num + 1):
                        # initialize the sub-figure
                        index = (row - 1) * col_num + col

                        # plot each material into the corresponding sub-figure
                        try:
                            fig_name = material_keys[index - 1]
                            ax = fig.add_subplot(row_num, col_num, index)
                            # plot the figure
                            self.curve_plotter.plot(
                                ax=ax,
                                material=materials[fig_name],
                                fig_name=fig_name,
                                x_stride=x_stride,
                                xlabel=xlabel,
                                ylabel=ylabel,
                            )
                        except IndexError:
                            pass

        # write to the tensorboard
        if tb_write and self.tb_writer is not None:
            for fig_name, material in materials.items():
                if isinstance(material, List):
                    self.tb_writer.add_scalar(
                        tag=os.path.join(*subfolder_names, fig_name),
                        scalar_value=material[-1],
                        global_step=epoch,
                    )
                elif isinstance(material, Dict):
                    if len(material) > 0:
                        self.tb_writer.add_scalars(
                            main_tag=os.path.join(*subfolder_names, fig_name),
                            tag_scalar_dict={
                                key: value[-1] for key, value in material.items()
                            },
                            global_step=epoch,
                        )

        # save all the information into files on disk for future usage
        if data_save:
            self.curve_plotter.save(
                materials=materials, save_path=save_path, x_stride=x_stride
            )

    def matrix_snapshot(
        self,
        save_path: str,
        subfolder_names: List[str],
        materials: Dict,
        epoch: int,
        xlabel: str,
        ylabel: str,
        sep_save: bool,
        sum_save: bool,
        tb_write: bool,
        data_save: bool,
        col_num: int,
        row_num: int,
        flip_y: bool = False,
    ):

        # save the plotted figure into a specific file
        if sep_save:
            # loop each file in the given material Dict
            for fig_name in materials.keys():
                with self.sep_figure_context(fig_name, save_path) as ax:
                    # plot the current material into a figure
                    self.matrix_plotter.plot(
                        ax=ax,
                        material=materials[fig_name],
                        fig_name=fig_name,
                        xlabel=xlabel,
                        ylabel=ylabel,
                        flip_y=flip_y,
                    )

        # save all the materials to a summary figure
        if sum_save:
            with self.sum_figure_context(
                materials, save_path, col_num, row_num, f"epoch{epoch}.png"
            ) as (fig, material_keys, row_num, col_num):
                # loop each row
                for row in range(1, row_num + 1):
                    # loop each column
                    for col in range(1, col_num + 1):
                        # initialize the sub-figure
                        index = (row - 1) * col_num + col

                        # plot each material into the corresponding sub-figure
                        try:
                            fig_name = material_keys[index - 1]
                            ax = fig.add_subplot(row_num, col_num, index)
                            # plot the figure
                            self.matrix_plotter.plot(
                                ax=ax,
                                material=materials[fig_name],
                                fig_name=fig_name,
                                xlabel=xlabel,
                                ylabel=ylabel,
                                flip_y=flip_y,
                            )
                        except IndexError:
                            pass

        # write the figure to the tensorboard
        if tb_write and self.tb_writer is not None:
            img = torchvision.io.read_image(
                os.path.join(save_path, f"epoch{epoch}.png")
            )
            self.tb_writer.add_image(
                tag=os.path.join(*subfolder_names), img_tensor=img, global_step=epoch
            )

        # save all the matrices into a single .npz file on disk for future usage
        if data_save:
            self.matrix_plotter.save(
                materials=materials, epoch=epoch, save_path=save_path
            )

    def hist_snapshot(self, save_path: str, materials: Dict, xlabel: str, ylabel: str):

        # loop each file in the given material Dict
        for fig_name in materials.keys():
            with self.sep_figure_context(fig_name, save_path) as ax:
                # plot the current material into a figure
                self.hist_plotter.plot(
                    ax=ax,
                    material=materials[fig_name],
                    fig_name=fig_name,
                    xlabel=xlabel,
                    ylabel=ylabel,
                )

    def text_snapshot(
        self,
        save_path: str,
        subfolder_names: List[str],
        materials: Dict[str, List[str]],
        epoch: int,
        data_save: bool,
        x_stride: int = 1,
    ):

        # loop each file in the given material Dict
        for file_name, material in materials.items():
            self.tb_writer.add_text(
                tag=os.path.join(*subfolder_names, file_name),
                text_string=material[-1],
                global_step=epoch,
            )

            if epoch is not None and data_save:
                x_axis = np.arange(0, len(material), dtype=np.int_) * x_stride + 1
                save_data = np.concatenate(
                    (x_axis.reshape(-1, 1), np.array(material).reshape(-1, 1)), axis=-1
                )

                # save each material into a specific .txt file for easy visualization
                np.savetxt(
                    f"{os.path.join(save_path, file_name)}.txt", save_data, fmt="%s"
                )

    def audio_snapshot(
        self,
        save_path: str,
        subfolder_names: List[str],
        materials: Dict[str, torch.Tensor],
        epoch: int,
        data_save: bool,
        sample_rate: int,
        audio_format: str,
    ):

        for file_name, material in materials.items():
            self.tb_writer.add_audio(
                tag=os.path.join(*subfolder_names, file_name),
                snd_tensor=material,
                global_step=epoch,
                sample_rate=sample_rate,
            )

            # save each audio into a specific file for easy visualization
            if epoch is not None and data_save:
                if material.is_cuda:
                    material = material.detach().cpu()
                sf.write(
                    file=f"{os.path.join(save_path, file_name)}.{audio_format}",
                    data=material.numpy(),
                    samplerate=sample_rate,
                    subtype=sf.default_subtype(audio_format.upper()),
                )
