from typing import Tuple, Optional, List
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from tqdm import tqdm
import os

def plot_input_output(
    u_k: np.ndarray,
    y_k: np.ndarray,
    u_s: np.ndarray,
    y_s: np.ndarray,
    initial_steps: Optional[int] = None,
    initial_text: str = "Init. Measurement",
    control_text: str = "Data-Driven MPC",
    display_initial_text: bool = True,
    display_control_text: bool = True,
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 300,
    u_ylimit: Optional[List[Tuple[float, float]]] = None,
    y_ylimit: Optional[List[Tuple[float, float]]] = None,
    fontsize: int = 12
) -> None:
    """
    Plot input-output data with setpoints in a Matplotlib figure.

    This function creates a 2 rows of subplots, with the first row containing
    control inputs, and the second row, system outputs. Each subplot shows the
    data series for each data sequence alongside its setpoint as a constant
    line.

    If provided, the first 'initial_steps' time steps can be highlighted to
    emphasize the initial input-output data measurement period representing
    the data-driven system characterization phase in a Data-Driven MPC
    algorithm. Additionally, custom labels can be displayed to indicate the
    initial measurement and the subsequent MPC control periods, but only if
    there is enough space to prevent them from overlapping with other plot
    elements.

    Args:
        u_k (np.ndarray): An array containing control input data of shape (T,
            m), where `m` is the number of inputs and `T` is the number of
            time steps.
        y_k (np.ndarray): An array containing system output data of shape (T,
            p), where `p` is the number of outputs and `T` is the number of
            time steps.
        u_s (np.ndarray): An array of shape (m, 1) containing `m` input
            setpoint values.
        y_s (np.ndarray): An array of shape (p, 1) containing `p` output
            setpoint values.
        initial_steps (Optional[int]): The number of initial time steps where
            input-output measurements were taken for the data-driven
            characterization of the system. This will highlight the initial
            measurement period in the plot.
        initial_text (str): Label text to display over the initial measurement
            period of the plot. Default is "Init. Measurement".
        control_text (str): Label text to display over the post-initial
            control period. Default is "Data-Driven MPC".
        display_initial_text (bool): Whether to display the `initial_text`
            label on the plot. Default is True.
        display_control_text (bool): Whether to display the `control_text`
            label on the plot. Default is True.
        figsize (Tuple[int, int]): The (width, height) dimensions of the
            created Matplotlib figure.
        dpi (int): The DPI resolution of the figure.
        u_ylimit (Optional[List[Tuple[float, float]]]): A list of tuples
            specifying the Y-axis limits for the input subplots.
        y_ylimit (Optional[List[Tuple[float, float]]]): A list of tuples
            specifying the Y-axis limits for the output subplots.
        fontsize (int): The fontsize for labels, legends and axes ticks.
    
    Raises:
        ValueError: If any array dimensions mismatch expected shapes or if the
            length of `u_ylimit` or `y_ylimit` does not match the number of
            subplots.
    """
    # Check input-output data dimensions
    if not (u_k.shape[0] == y_k.shape[0]):
        raise ValueError("Dimension mismatch. The number of time steps for "
                         "u_k and y_k do not match.")
    if not (u_k.shape[1] == u_s.shape[0] and y_k.shape[1] == y_s.shape[0]):
        raise ValueError("Dimension mismatch. The number of inputs from u_k "
                         "and u_s, and the number of outputs from y_k and "
                         "y_s should match.")
    
    # Retrieve number of input and output data sequences and their length
    m = u_k.shape[1] # Number of inputs
    p = y_k.shape[1] # Number of outputs

    # Error handling for y-limit lengths
    if u_ylimit and len(u_ylimit) != u_k.shape[1]:
        raise ValueError(f"The length of `u_ylimit` ({len(u_ylimit)}) does "
                         f"not match the number of input subplots ({m}).")
    if y_ylimit and len(y_ylimit) != p:
        raise ValueError(f"The length of `y_ylimit` ({len(y_ylimit)}) does "
                         f"not match the number of output subplots ({p}).")
    
    # Create figure
    fig = plt.figure(layout='constrained', figsize=figsize, dpi=dpi)
    
    # Modify constrained layout padding
    fig.set_constrained_layout_pads(
        w_pad=0.1, h_pad=0.1, wspace=0.05, hspace=0)

    # Create subfigures for input and output data plots
    subfigs = fig.subfigures(2, 1)

    # Add titles for input and output subfigures
    subfigs[0].suptitle('Control Inputs',
                        fontsize=fontsize + 2,
                        fontweight='bold')
    subfigs[1].suptitle('System Outputs',
                        fontsize=fontsize + 2,
                        fontweight='bold')

    # Create subplots
    axs_u = subfigs[0].subplots(1, max(m, p))
    axs_y = subfigs[1].subplots(1, max(m, p))

    # Plot input data
    for i in range(m):
        plot_data(axis=axs_u[i],
                  data=u_k[:, i],
                  setpoint=u_s[i, :],
                  index=i,
                  var_symbol="u",
                  var_label="Input",
                  initial_steps=initial_steps,
                  initial_text=initial_text,
                  control_text=control_text,
                  display_initial_text=display_initial_text,
                  display_control_text=display_control_text,
                  ylimit=u_ylimit,
                  fontsize=fontsize,
                  fig=fig)

    # Plot output data
    for j in range(p):
        plot_data(axis=axs_y[j],
                  data=y_k[:, j],
                  setpoint=y_s[j, :],
                  index=j,
                  var_symbol="y",
                  var_label="Output",
                  initial_steps=initial_steps,
                  initial_text=initial_text,
                  control_text=control_text,
                  display_initial_text=display_initial_text,
                  display_control_text=display_control_text,
                  ylimit=y_ylimit,
                  fontsize=fontsize,
                  fig=fig)

    # Show the plot
    plt.show()

def plot_data(
    axis: Axes,
    data: np.ndarray,
    setpoint: float,
    index: int,
    var_symbol: str,
    var_label: str,
    initial_steps: Optional[int],
    initial_text: str,
    control_text: str,
    display_initial_text: bool,
    display_control_text: bool,
    ylimit: Optional[List[Tuple[float, float]]],
    fontsize: int,
    fig: Figure
) -> None:
    """
    Plot a data series with setpoints in a specified axis. Optionally,
    highlight an initial measurement phase and a control phase using shaded
    regions and text labels. The labels will be displayed if there is enough
    space to prevent them from overlapping with other plot elements.

    Args:
        axis (Axes): The Matplotlib axis object to plot on.
        data (np.ndarray): An array containing data to be plotted.
        setpoint (float): The setpoint value for the data.
        index (int): The index of the data used for labeling purposes (e.g.,
            "u_1", "u_2").
        var_symbol (str): The variable symbol used to label the data series
            (e.g., "u" for inputs, "y" for outputs).
        var_label (str): The variable label representing the control signal
            (e.g., "Input", "Output").
        initial_steps (Optional[int]): The number of initial time steps where
            input-output measurements were taken for the data-driven
            characterization of the system. This will highlight the initial
            measurement period in the plot.
        initial_text (str): Label text to display over the initial measurement
            period of the plot.
        control_text (str): Label text to display over the post-initial
            control period.
        display_initial_text (bool): Whether to display the `initial_text`
            label on the plot.
        display_control_text (bool): Whether to display the `control_text`
            label on the plot.
        ylimit (Optional[Tuple[float, float]]): A tuple specifying the Y-axis
            limits for the plot.
        fontsize (int): The fontsize for labels, legends and axes ticks.
        fig (Figure): The Matplotlib figure object containing the axis.
    """
    T = data.shape[0] # Data length

    # Plot data series
    axis.plot(range(0, T), data, label=f'${var_symbol}_{index + 1}$')
    # Plot setpoint
    axis.plot(range(0, T), np.full(T, setpoint),
              label=f'${var_symbol}_{index + 1}^s$')
    
    # Highlight initial input-output data measurement period if provided
    if initial_steps:
        # Highlight period with a grayed rectangle
        axis.axvspan(0, initial_steps, color='gray', alpha=0.1)
        # Add a vertical line at the right side of the rectangle
        axis.axvline(x=initial_steps, color='black',
                            linestyle=(0, (5, 5)), linewidth=1)
        
        # Display initial measurement text if enabled
        if display_initial_text:
            # Get y-axis limits
            y_min, y_max = axis.get_ylim()
            # Place label at the center of the highlighted area
            u_init_text = axis.text(
                initial_steps / 2, (y_min + y_max) / 2,
                initial_text, fontsize=fontsize - 1,
                ha='center', va='center', color='black',
                bbox=dict(facecolor='white', edgecolor='black'))
            # Get initial text bounding box width
            init_text_width = get_text_width_in_data(
                text_object=u_init_text, axis=axis, fig=fig)
            # Hide text box if it overflows the plot area
            if initial_steps < init_text_width:
                u_init_text.set_visible(False)
        
        # Display Data-Driven MPC control text if enabled
        if display_control_text:
            # Get y-axis limits
            y_min, y_max = axis.get_ylim()
            # Place label at the center of the remaining area
            u_control_text = axis.text(
                (T + initial_steps) / 2, (y_min + y_max) / 2,
                control_text, fontsize=fontsize - 1,
                ha='center', va='center', color='black',
                bbox=dict(facecolor='white', edgecolor='black'))
            # Get control text bounding box width
            control_text_width = get_text_width_in_data(
                text_object=u_control_text, axis=axis, fig=fig)
            # Hide text box if it overflows the plot area
            if (T - initial_steps) < control_text_width:
                u_control_text.set_visible(False)
    
    # Format labels, legend and ticks
    axis.set_xlabel('Time step $k$', fontsize=fontsize)
    axis.set_ylabel(f'{var_label} ${var_symbol}_{index + 1}$',
                    fontsize=fontsize)
    axis.legend(fontsize=fontsize)
    axis.tick_params(axis='both', labelsize=fontsize)
    
    # Set x-limits
    axis.set_xlim([0, T])

    # Set y-limits if provided
    if ylimit:
        axis.set_ylim(ylimit)

def plot_input_output_animation(
    u_k: np.ndarray,
    y_k: np.ndarray,
    u_s: np.ndarray,
    y_s: np.ndarray,
    initial_steps: Optional[int] = None,
    initial_text: str = "Init. Measurement",
    control_text: str = "Data-Driven MPC",
    display_initial_text: bool = True,
    display_control_text: bool = True,
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 300,
    interval: int = 10,
    fontsize: int = 12
) -> FuncAnimation:
    """
    Create a Matplotlib animation showing the progression of input-output data
    over time.

    This function generates a figure with two rows of subplots: the top
    subplots display control inputs and the bottom subplots display system
    outputs. Each subplot shows the data series for each sequence alongside
    its setpoint as a constant line.

    If provided, the first 'initial_steps' time steps can be highlighted to
    emphasize the initial input-output data measurement period representing
    the data-driven system characterization phase in a Data-Driven MPC
    algorithm. Additionally, custom labels can be displayed to indicate the
    initial measurement and the subsequent MPC control periods, but only if
    there is enough space to prevent them from overlapping with other plot
    elements.
    
    Args:
        u_k (np.ndarray): An array containing control input data of shape (T,
            m), where `m` is the number of inputs and `T` is the number of
            time steps.
        y_k (np.ndarray): An array containing system output data of shape (T,
            p), where `p` is the number of outputs and `T` is the number of
            time steps.
        u_s (np.ndarray): An array of shape (m, 1) containing `m` input
            setpoint values.
        y_s (np.ndarray): An array of shape (p, 1) containing `p` output
            setpoint values.
        initial_steps (Optional[int]): The number of initial time steps where
            input-output measurements were taken for the data-driven
            characterization of the system. This will highlight the initial
            measurement period in the plot.
        initial_text (str): Label text to display over the initial measurement
            period of the plot. Default is "Init. Measurement".
        control_text (str): Label text to display over the post-initial
            control period. Default is "Data-Driven MPC".
        display_initial_text (bool): Whether to display the `initial_text`
            label on the plot. Default is True.
        display_control_text (bool): Whether to display the `control_text`
            label on the plot. Default is True.
        figsize (Tuple[int, int]): The (width, height) dimensions of the
            created Matplotlib figure.
        dpi (int): The DPI resolution of the figure.
        interval (int): The time between frames in milliseconds.
        fontsize (int): The fontsize for labels, legends and axes ticks.
    """
    # Check input-output data dimensions
    if not (u_k.shape[0] == y_k.shape[0]):
        raise ValueError("Dimension mismatch. The number of time steps for "
                         "u_k and y_k do not match.")
    if not (u_k.shape[1] == u_s.shape[0] and y_k.shape[1] == y_s.shape[0]):
        raise ValueError("Dimension mismatch. The number of inputs from u_k "
                         "and u_s, and the number of outputs from y_k and "
                         "y_s should match.")
    
    # Retrieve number of input and output data sequences and their length
    m = u_k.shape[1] # Number of inputs
    p = y_k.shape[1] # Number of outputs
    T = u_k.shape[0] # Length of data

    # Create figure
    fig = plt.figure(layout='constrained', figsize=figsize, dpi=dpi)
    
    # Modify constrained layout padding
    fig.set_constrained_layout_pads(
        w_pad=0.1, h_pad=0.1, wspace=0.05, hspace=0)

    # Create subfigures for input and output data plots
    subfigs = fig.subfigures(2, 1)

    # Add titles for input and output subfigures
    subfigs[0].suptitle('Control Inputs',
                        fontsize=fontsize + 2,
                        fontweight='bold')
    subfigs[1].suptitle('System Outputs',
                        fontsize=fontsize + 2,
                        fontweight='bold')
    
    # Create subplots
    axs_u = subfigs[0].subplots(1, max(m, p))
    axs_y = subfigs[1].subplots(1, max(m, p))

    # Define input-output line lists
    u_lines: List[Line2D] = []
    y_lines: List[Line2D] = []

    # Define initial measurement rectangles and texts lists
    u_rects: List[Rectangle] = []
    u_rect_lines: List[Line2D] = []
    u_init_texts: List[Text] = []
    u_control_texts: List[Text] = []
    y_rects: List[Rectangle] = []
    y_rect_lines: List[Line2D] = []
    y_init_texts: List[Text] = []
    y_control_texts: List[Text] = []

    # Define y-axis center
    u_y_axis_centers: List[float] = []
    y_y_axis_centers: List[float] = []
        
    # Initialize input plot elements
    for i in range(m):
        initialize_data_animation(axis=axs_u[i],
                                  data=u_k[:, i],
                                  setpoint=u_s[i, :],
                                  index=i,
                                  var_symbol="u",
                                  var_label="Input",
                                  initial_steps=initial_steps,
                                  initial_text=initial_text,
                                  control_text=control_text,
                                  fontsize=fontsize,
                                  lines=u_lines,
                                  rects=u_rects,
                                  rect_lines=u_rect_lines,
                                  init_texts=u_init_texts,
                                  control_texts=u_control_texts,
                                  y_axis_centers=u_y_axis_centers)
    
    # Initialize output plot elements
    for j in range(p):
        initialize_data_animation(axis=axs_y[j],
                                  data=y_k[:, j],
                                  setpoint=y_s[j, :],
                                  index=j,
                                  var_symbol="y",
                                  var_label="Output",
                                  initial_steps=initial_steps,
                                  initial_text=initial_text,
                                  control_text=control_text,
                                  fontsize=fontsize,
                                  lines=y_lines,
                                  rects=y_rects,
                                  rect_lines=y_rect_lines,
                                  init_texts=y_init_texts,
                                  control_texts=y_control_texts,
                                  y_axis_centers=y_y_axis_centers)
    
    # Get initial text bounding box width
    init_text_width = get_text_width_in_data(
        text_object=u_init_texts[0], axis=axs_u[0], fig=fig)
    # Get control text bounding box width
    control_text_width = get_text_width_in_data(
        text_object=u_control_texts[0], axis=axs_u[0], fig=fig)

    # Animation update function
    def update(frame):
        # Update input plot data
        for i in range(m):
            update_data_animation(frame=frame,
                                  data=u_k[:frame+1, i],
                                  data_length=T,
                                  initial_steps=initial_steps,
                                  line=u_lines[i],
                                  rect=u_rects[i],
                                  y_axis_center=u_y_axis_centers[i],
                                  rect_line=u_rect_lines[i],
                                  init_text_obj=u_init_texts[i],
                                  control_text_obj=u_control_texts[i],
                                  display_initial_text=display_initial_text,
                                  display_control_text=display_control_text,
                                  init_text_width=init_text_width,
                                  control_text_width=control_text_width)
        
        # Update output plot data
        for j in range(p):
            update_data_animation(frame=frame,
                                  data=y_k[:frame+1, j],
                                  data_length=T,
                                  initial_steps=initial_steps,
                                  line=y_lines[j],
                                  rect=y_rects[j],
                                  y_axis_center=y_y_axis_centers[j],
                                  rect_line=y_rect_lines[j],
                                  init_text_obj=y_init_texts[j],
                                  control_text_obj=y_control_texts[j],
                                  display_initial_text=display_initial_text,
                                  display_control_text=display_control_text,
                                  init_text_width=init_text_width,
                                  control_text_width=control_text_width)

        return (u_lines + y_lines + u_rects + u_rect_lines +
                u_init_texts + u_control_texts + y_rects +
                y_init_texts + y_control_texts + y_rect_lines)

    # Create animation
    animation = FuncAnimation(
        fig, update, frames=T, interval=interval, blit=True)

    return animation

def initialize_data_animation(
    axis: Axes,
    data: np.ndarray,
    setpoint: float,
    index: int,
    var_symbol: str,
    var_label: str,
    initial_steps: Optional[int],
    initial_text: str,
    control_text: str,
    fontsize: int,
    lines: List[Line2D],
    rects: List[Rectangle],
    rect_lines: List[Line2D],
    init_texts: List[Text],
    control_texts: List[Text],
    y_axis_centers: List[float]
) -> None:
    """
    Initialize plot elements for a data series animation with setpoints.

    This function initializes and appends several elements to the plot, such
    as plot lines representing data, rectangles and lines representing an
    initial input-output data measurement period, and text labels for both the
    initial measurement and control periods. It also adjusts the axis limits
    and stores the y-axis center values.

    Args:
        axis (Axes): The Matplotlib axis object to plot on.
        data (np.ndarray): An array containing data to be plotted.
        setpoint (float): The setpoint value for the data.
        index (int): The index of the data used for labeling purposes (e.g.,
            "u_1", "u_2").
        var_symbol (str): The variable symbol used to label the data series
            (e.g., "u" for inputs, "y" for outputs).
        var_label (str): The variable label representing the control signal
            (e.g., "Input", "Output").
        initial_steps (Optional[int]): The number of initial time steps where
            input-output measurements were taken for the data-driven
            characterization of the system. This will highlight the initial
            measurement period in the plot.
        initial_text (str): Label text to display over the initial measurement
            period of the plot.
        control_text (str): Label text to display over the post-initial
            control period.
        fontsize (int): The fontsize for labels, legends and axes ticks.
        lines (List[Line2D]): The list where the initialized plot lines will
            be stored.
        rects (List[Rectangle]): The list where the initialized rectangles
            representing the initial measurement region will be stored.
        rect_lines (List[Line2D]): The list where the initialized vertical
            lines representing the initial measurement region limit will be
            stored.
        init_texts (List[Text]): The list where the initialized initial
            measurement label texts will be stored.
        control_texts (List[Text]): The list where the initialized control
            label texts will be stored.
        y_axis_centers (List[float]): The list where the y-axis center from
            the adjusted axis will be stored.
    
    Note:
        This function updates the `lines`, `rects`, `rect_lines`,
        `init_texts`, and `control_texts` with the initialized plot elements.
        It also adjusts the y-axis limits to a fixed range and stores the
        center values in `y_axis_centers`.
    """
    T = data.shape[0] # Data length

    # Initialize plot lines
    lines.append(axis.plot([], [], label=f'${var_symbol}_{index + 1}$')[0])
    # Plot setpoint
    axis.plot(
        range(0, T), np.full(T, setpoint), label=f'${var_symbol}_{index + 1}^s$')
    # Format labels, legend and ticks
    axis.set_xlabel('Time step $k$', fontsize=fontsize)
    axis.set_ylabel(f'{var_label} ${var_symbol}_{index + 1}$', fontsize=fontsize)
    axis.legend(fontsize=fontsize)
    axis.tick_params(axis='both', labelsize=fontsize)

    # Define axis limits
    u_lim_min, u_lim_max = get_padded_limits(data, setpoint)
    axis.set_xlim(0, T)
    axis.set_ylim(u_lim_min, u_lim_max)
    y_axis_centers.append((u_lim_min + u_lim_max) / 2)

    if initial_steps:
        # Initialize initial measurement rectangle
        rects.append(axis.axvspan(0, 0, color='gray', alpha=0.1))
        # Initialize initial measurement rectangle limit line
        rect_lines.append(axis.axvline(
            x=0, color='black', linestyle=(0, (5, 5)), linewidth=1))
        # Initialize initial measurement text
        init_texts.append(axis.text(
            initial_steps / 2, y_axis_centers[index],
            initial_text, fontsize=fontsize - 1, ha='center',
            va='center', color='black', bbox=dict(facecolor='white',
                                                    edgecolor='black')))
        # Initialize control text
        control_texts.append(axis.text(
            (T + initial_steps) / 2, y_axis_centers[index],
            control_text, fontsize=fontsize - 1, ha='center',
            va='center', color='black', bbox=dict(facecolor='white',
                                                    edgecolor='black')))

def update_data_animation(
    frame: int,
    data: np.ndarray,
    data_length: int,
    initial_steps: Optional[int],
    line: Line2D,
    rect: Rectangle,
    y_axis_center: float,
    rect_line: Line2D,
    init_text_obj: Text,
    control_text_obj: Text,
    display_initial_text: bool,
    display_control_text: bool,
    init_text_width: float,
    control_text_width: float
) -> None:
    """
    Update the plot elements in a data series animation with setpoints.

    This function updates the data plot line for each animation frame.
    If 'initial_steps' is provided, it also updates the rectangle and line
    representing the initial input-output measurement period, as well as the
    text labels indicating the initial measurement and control periods. These
    labels will be displayed if there is enough space to prevent them from
    overlapping with other plot elements.

    Args:
        frame (int): The current animation frame.
        data (np.ndarray): An array containing data to be plotted.
        data_length (int): The length of the `data` array.
        initial_steps (Optional[int]): The number of initial time steps where
            input-output measurements were taken for the data-driven
            characterization of the system. This will highlight the initial
            measurement period in the plot.
        line (Line2D): The plot line corresponding to the data series plot.
        rect (Rectangle): The rectangle representing the initial measurement
            region.
        y_axis_center (float): The y-axis center of the plot axis.
        rect_line (Line2D): The line object representing the initial
            measurement region limit.
        init_text_obj (Text): The text object containing the initial
            measurement period label.
        control_text_obj (Text): The text object containing the control period
            label.
        display_initial_text (bool): Whether to display the `initial_text`
            label on the plot.
        display_control_text (bool): Whether to display the `control_text`
            label on the plot.
        init_text_width (float): The width of the `init_text_obj` object in
            data coordinates.
        control_text_width (float): The width of the `control_text_obj` object
            in data coordinates.
    """
    # Update plot line data
    line.set_data(range(0, frame+1), data[:frame+1])
    
    # Update initial measurement rectangle and texts
    if initial_steps and frame <= initial_steps:
        # Update rectangle width
        rect.set_width(frame)
        # Hide initial measurement and control texts
        init_text_obj.set_visible(False)
        control_text_obj.set_visible(False)
        # Update rectangle limit line position
        rect_line.set_xdata([frame])
        # Show initial measurement text
        if display_initial_text and frame >= init_text_width:
            init_text_obj.set_position((frame / 2, y_axis_center))
            init_text_obj.set_visible(True)
        # Show control text if possible
        if display_control_text and frame == initial_steps:
            if (data_length - initial_steps) >= control_text_width:
                control_text_obj.set_visible(True)

def save_animation(
    animation: FuncAnimation,
    total_frames: int,
    fps: int,
    bitrate: int,
    file_path: str
) -> None:
    """
    Save a Matplotlib animation using an ffmpeg writer with progress bar
    tracking.

    This function saves the given Matplotlib animation to the specified file
    path and displays a progress bar the console to track the saving progress.
    If the file path contains directories that do not exist, they will be
    created.

    Args:
        animation (FuncAnimation): The animation object to save.
        total_frames (int): The total number of frames in the animation.
        fps (int): The frames per second of saved video.
        bitrate (int): The bitrate of saved video.
        file_path (str): The path (including directory and file name) where 
            the animation will be saved.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Set up the ffmpeg writer
    writer = FFMpegWriter(fps=fps,
                          metadata=dict(artist='Me'),
                          bitrate=bitrate)
    
    # Save animation while displaying a progress bar
    with tqdm(total=total_frames, desc="Saving animation") as pbar:
        animation.save(file_path,
                       writer=writer,
                       progress_callback=lambda i, n: pbar.update(1))

def get_padded_limits(
    X: np.ndarray,
    X_s: np.ndarray,
    pad_percentage: float = 0.05
) -> Tuple[float, float]:
    """
    Get the minimun and maximum limits from two data sequences extended by
    a specified percentage of the combined data range.

    Args:
        X (np.ndarray): First data array.
        X_s (np.ndarray): Second data array.
        pad_percentage (float, optional): The percentage of the data range
            to be used as padding. Defaults to 0.05.

    Returns:
        Tuple[float, float]: A tuple containing padded minimum and maximum 
            limits for the combined data from `X` and `X_s`.
    """
    # Get minimum and maximum limits from data sequences
    X_min, X_max = np.min(X), np.max(X)
    X_s_min, X_s_max = np.min(X_s), np.max(X_s)
    X_lim_min = min(X_min, X_s_min)
    X_lim_max = max(X_max, X_s_max)

    # Extend limits by a percentage of the overall data range
    X_range = X_lim_max - X_lim_min
    X_lim_min -= X_range * pad_percentage
    X_lim_max += X_range * pad_percentage

    return (X_lim_min, X_lim_max)

def get_text_width_in_data(
    text_object: Text,
    axis: Axes,
    fig: Figure
) -> float:
    """
    Calculate the bounding box width of a text object in data coordinates.

    Args:
        text_object (Text): A Matplotlib text object.
        axis (Axes): The axis on which the text object is displayed.
        fig (Figure): The Matplotlib figure object containing the axis.

    Returns:
        float: The width of the text object's bounding box in data
            coordinates.
    """
    # Get the bounding box of the text object in pixel coordinates
    text_box = text_object.get_window_extent(
        renderer=fig.canvas.get_renderer())
    # Convert the bounding box from pixel coordinates to data coordinates
    text_box_data = axis.transData.inverted().transform(text_box)
    # Calculate the width of the bounding box in data coordinates
    text_box_width = text_box_data[1][0] - text_box_data[0][0]
    
    return text_box_width