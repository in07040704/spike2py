from imaplib import Int2AP
from typing import Literal, Tuple

import numpy as np
import matplotlib
from matplotlib.axes._subplots import Subplot

import matplotlib.pyplot as plt


from spike2py import channels, trial
from spike2py.types import all_channels, ticksline_channels

LINE_WIDTH = 2
FIG_SIZE = (12, 4)
WAVEFORM_FIG_SIZE = (12, 8)
MAX_TRIAL_FIG_HEIGHT = 30
LEGEND_LOC = "upper right"
COLORS = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
]

matplotlib.rcParams.update({"font.size": 14})


def my_proc_channel(spike2py_channel: all_channels, save: Literal[True, False]) -> None:
    """Plot individual channels.

    Parameters
    ----------
    spike2py_channel:
        Instance of spike2py.channels.<ch> where possible ch are
        Event, Keyboard, Wavemark and Waveform
    save:
        Whether or not to save the generated figure.

    Returns
    -------
    None
    """

    if len(spike2py_channel.times) == 0:
        print("{spike2py_channel.info.name} channel has no data to process.")
        return
    channel_type = repr(spike2py_channel).split()[0]
    if channel_type == "Waveform":
        # Split axis generation and plotting to allow reuse of plotting with trial plotting
        #fig, ax = plt.subplots(figsize=WAVEFORM_FIG_SIZE)
        _proc_waveform(spike2py_channel)
    else:
        ticks_line = _TicksLine(spike2py_channel)
        if ticks_line.ch_type == "Wavemark":
            # fig, ax = plt.subplots(
            #     1, 2, figsize=FIG_SIZE, gridspec_kw={"width_ratios": [3, 1]}
            # )
            # ticks_line.plot(ax)
            ticks_line.myproc()
        else:
            # fig, ax = plt.subplots(figsize=FIG_SIZE)
            # ticks_line.plot(ax)
            ticks_line.myproc()

    if save:
        _save_plot(spike2py_channel.info)


def _get_color(index: int) -> str:
    return COLORS[index % len(COLORS)]


def _proc_waveform(
    waveform: "channels.Waveform"
    #ax: Subplot,
    #color: str = _get_color(0),
) -> None:
    print("WFtime:")
    print(waveform.times)
    print("WFvalues:")
    print(waveform.values)

def _proc_waveform2(
    waveform: "channels.Waveform",
    dic1: dict,
    dic2: dict,
    threshold: float
    #ax: Subplot,
    #color: str = _get_color(0),
) -> None:
    #waveform.times, waveform.values
    v_data=waveform.values
    t_data=waveform.times
    v_data_1=v_data > threshold
    t_data_2=t_data[v_data_1]

    s_time_dict={}
    diff_t_data = np.diff(t_data_2)
    diff_t_data1=diff_t_data > 0.3
    diff_t_data2=np.insert(diff_t_data1, 0, False)
    s_time=t_data_2[diff_t_data2]
    s_time=np.insert(s_time, 0, t_data_2[0])

    diff_t_data3=np.insert(diff_t_data1, -1, False)
    s_time2=t_data_2[diff_t_data3]
    s_time2=np.append(s_time2,t_data_2[-1])

    max_list=[]
    for ti_n1, ti_n2 in zip(s_time, s_time2):
        bin_time = [(t_data >= ti_n1) & (t_data < ti_n2)]
        bin_volte = v_data[bin_time]
        max_volte =bin_volte.max()
        max_list.append(max_volte)


    s_time_dict["codes"] = [f'w{q}' for q in range(len(s_time))]
    s_time_dict["times"] = s_time
    s_time_dict["max_volte"] = max_list
    print("W_n_dic:")
    print(s_time_dict)    




def _save_plot(channel_info: "channels.ChannelInfo") -> None:
    fig_name = (
        f"{channel_info.subject_id}_"
        f"{channel_info.trial_name}_"
        f"{channel_info.name}.png"
    )
    fig_path = channel_info.path_save_figures / fig_name
    plt.savefig(fig_path)
    plt.close()


class _TicksLine:
    """Class that manages plotting of Event, Keyboard and Wavemark channels"""

    def __init__(
        self,
        ticks_line_channel: ticksline_channels,
    ):
        """Initialise TicksLine Class

        Parameters
        ----------
        ticks_line_channel:
            Instance of spike2py.channels.< > where possible channel types are Event, Keyboard, and Wavemark
        color:
            Set color of ticks and line
        y_offset:
            Offset of ticks and line in the y-direction
        """

        self.ch = ticks_line_channel
        # self.color = color
        # self.offset = y_offset

        self.ch_type = repr(ticks_line_channel).split()[0]
        self.line_start_end = (self.ch.times[0], self.ch.times[-1])
        # self.line_y_vals = (0.5 + y_offset, 0.5 + y_offset)
        # self.tick_y_vals = (0.2 + y_offset, 0.8 + y_offset)

    def myproc(self):
        # if isinstance(ax, np.ndarray):
        #     ax1 = ax[0]
        #     ax2 = ax[1]
        # else:
        #     ax1 = ax
        #     ax2 = None

        # self._plot_ticks_line(ax1)
        # self._finalise_plot(ax1)

        if self.ch_type == "Keyboard":
            s_code_dic, w_code_dic = self._proc_codes()
        if (self.ch_type == "Wavemark"):
            self._plot_action_potentials()
        
        return s_code_dic, w_code_dic

    #    plt.tight_layout()

    # def _plot_ticks_line(self, ax1: Subplot):
    #     for time in self.ch.times:
    #         ax1.plot(
    #             (time, time), self.tick_y_vals, linewidth=LINE_WIDTH, color=self.color
    #         )
    #     ax1.plot(
    #         self.line_start_end,
    #         self.line_y_vals,
    #         linewidth=LINE_WIDTH,
    #         label=self.ch.info.name,
    #         color=self.color,
    #     )

    def _proc_codes(self):
        i=0
        i_s=0
        i_w=0
        s_code_list=[]
        s_time_list=[]
        w_code_list=[]
        w_time_list=[]
        s_dict={}
        w_dict={}
        for time, code in zip(self.ch.times, self.ch.codes):
            i=i+1
            if code =="S":
                s_code_list.append(i_s)
                s_time_list.append(time)
                i_s=i_s+1
            elif code =="w":
                w_code_list.append(i_w)
                w_time_list.append(time)
                i_w=i_w+1
            else:
                print("Error")
                print(f'{code} {i}/{time}')
        
        arr_s_time = np.array(s_time_list, dtype=float)
        diff_s_time = np.diff(arr_s_time)
        diff_s_time1=diff_s_time>5
        diff_s_time1=np.insert(diff_s_time1, 0, False)
        s_code=arr_s_time[diff_s_time1]
        s_code=np.insert(s_code, 0, arr_s_time[0])
        s_dict["codes"] = [f'S{q}' for q in range(len(s_code))]
        s_dict["times"] = s_code
        print(s_dict)

        arr_w_time = np.array(w_time_list, dtype=float)
        diff_w_time = np.diff(arr_w_time)
        diff_w_time1=diff_w_time>20
        diff_w_time1=np.insert(diff_w_time1, 0, False)
        w_code=arr_w_time[diff_w_time1]
        w_code=np.insert(w_code, 0, arr_w_time[0])
        w_dict["codes"] = [f'W{q}' for q in range(len(w_code))]
        w_dict["times"] = w_code
        print(w_dict)

        return s_dict, w_dict




            # print(time)
            # print(f'{code} {i}/')
            # if code =="s":
            #     i_s=1+i_s
            #     if i_s==1:
            #         i_s2=1+i_s2
            #         s_sec_list.append(f'{code}{i_s2}')
            #     else:


                
            # elif code =="w":

            # else:
            #     pass
            


    def _plot_action_potentials(self):
        for action_potential in self.ch.action_potentials:
            print("WMx:")
            print(action_potential)
        # ax2.get_yaxis().set_visible(False)
        # ax2.get_xaxis().set_visible(False)

    # def _finalise_plot(self, ax1: Subplot):
    #     ax1.legend(loc=LEGEND_LOC)
    #     ax1.set_xlabel("time (s)")
    #     ax1.get_yaxis().set_visible(False)
    #     ax1.grid()


def proc_trial(spike2py_trial: "trial.Trial", save: Literal[True, False]) -> None:
    # fig_height, n_subplots = _proc_n_sub(spike2py_trial)
    # if n_subplots == 1:
    #     print(
    #         f"The trial `{spike2py_trial.name}` has only one plottable channel."
    #         "\nPlease use `trial_name.ch_name.plot()` instead."
    #     )
    # fig, ax = plt.subplots(
    #     sharex=True,
    #     nrows=n_subplots,
    #     figsize=(12, fig_height),
    #     gridspec_kw={"hspace": 0},
    # )
    print("Are you")
    _proc_trial(spike2py_trial)
    if save:
        _save_plot(spike2py_trial.name)


def _proc_n_sub(spike2py_trial: "trial.Trial"):
    """Determine height and number of subplots to plot trial.

    Event, Keyboard and Wavemark channels are all plotted on same subplot at the top of the figure.
    Need to make sure these channels have data."""
    fig_height = 4
    n_subplots = 0
    plottable_ticks_line = False
    for channel, channel_type in spike2py_trial.channels:
        current_channel = spike2py_trial.__getattribute__(channel)
        if (
            (channel_type in ["event", "keyboard", "wavemark"])
            and (not plottable_ticks_line)
            and (len(current_channel.times) != 0)
        ):
            fig_height += 2
            plottable_ticks_line = True
            n_subplots += 1
        elif channel_type == "waveform":
            fig_height += 2
            n_subplots += 1
    fig_height = min(fig_height, MAX_TRIAL_FIG_HEIGHT)
    return fig_height, n_subplots


def _proc_trial(spike2py_trial: "trial.Trial"):
    print("hear?")
    print(spike2py_trial)
    print(spike2py_trial.channels)
    for channel, channel_type in spike2py_trial.channels:
        if channel_type == "keyboard":
            current_channel = spike2py_trial.__getattribute__(channel)
            ticks_line = _TicksLine(
                ticks_line_channel=current_channel,
            )
            s_dic, w_dic = ticks_line.myproc()
            print("I am")

        elif channel == "Myhy_R":
            current_channel = spike2py_trial.__getattribute__(channel)
            _proc_waveform2(
                waveform=current_channel,
                dic1=s_dic, 
                dic2=w_dic,
                threshold=0.01
            )
            print(" Hear.")
        else:
            pass
