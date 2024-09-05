from typing import Tuple, Optional, List
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
    there is sufficient space to prevent them from overflowing the plots.

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
    T = u_k.shape[0] # Length of data

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

    # Plot data
    for i in range(m):
        # Plot input data
        axs_u[i].plot(range(0, T), u_k[:, i], label=f'$u_{i+1}$')
        # Plot input setpoint
        axs_u[i].plot(
            range(0, T), np.full(T, u_s[i, :]), label=f'$u_{i+1}^s$')
        
        # Highlight initial input-output data measurement period if provided
        if initial_steps:
            # Highlight period with a grayed rectangle
            axs_u[i].axvspan(0, initial_steps, color='gray', alpha=0.1)
            # Add a vertical line at the right side of the rectangle
            axs_u[i].axvline(x=initial_steps, color='black',
                             linestyle=(0, (5, 5)), linewidth=1)
            
            # Display initial measurement text if enabled
            if display_initial_text:
                # Get y-axis limits
                y_min, y_max = axs_u[i].get_ylim()
                # Place label at the center of the highlighted area
                u_init_text = axs_u[i].text(
                    initial_steps / 2, (y_min + y_max) / 2,
                    initial_text, fontsize=fontsize - 1,
                    ha='center', va='center', color='black',
                    bbox=dict(facecolor='white', edgecolor='black'))
                # Get initial text bounding box width
                init_text_width = get_text_width_in_data(
                    text_object=u_init_text, axis=axs_u[i], fig=fig)
                # Hide text box if it overflows the plot area
                if initial_steps < init_text_width:
                    u_init_text.set_visible(False)
            
            # Display Data-Driven MPC control text if enabled
            if display_control_text:
                # Get y-axis limits
                y_min, y_max = axs_u[i].get_ylim()
                # Place label at the center of the remaining area
                u_control_text = axs_u[i].text(
                    (T + initial_steps) / 2, (y_min + y_max) / 2,
                    control_text, fontsize=fontsize - 1,
                    ha='center', va='center', color='black',
                    bbox=dict(facecolor='white', edgecolor='black'))
                # Get control text bounding box width
                control_text_width = get_text_width_in_data(
                    text_object=u_control_text, axis=axs_u[0], fig=fig)
                # Hide text box if it overflows the plot area
                if (T - initial_steps) < control_text_width:
                    u_control_text.set_visible(False)
            
        axs_u[i].set_xlabel('Time step $k$', fontsize=fontsize)
        axs_u[i].set_ylabel(f'Input $u_{i+1}$', fontsize=fontsize)
        axs_u[i].legend(fontsize=fontsize)
        axs_u[i].set_xlim([0, T])
        axs_u[i].tick_params(axis='both', labelsize=fontsize)

        # Set y-limits if provided
        if u_ylimit and u_ylimit[i]:
            axs_u[i].set_ylim(u_ylimit[i])

    for j in range(p):
        # Plot output data
        axs_y[j].plot(range(0, T), y_k[:, j], label=f'$y_{j+1}$')
        # Plot output setpoint
        axs_y[j].plot(
            range(0, T), np.full(T, y_s[j, :]), label=f'$y_{j+1}^s$')
        
        # Highlight initial input-output data measurement period if provided
        if initial_steps:
            # Highlight period with a grayed rectangle
            axs_y[j].axvspan(0, initial_steps, color='gray', alpha=0.1)
            # Add a vertical line at the right side of the rectangle
            axs_y[j].axvline(x=initial_steps, color='black',
                             linestyle=(0, (5, 5)), linewidth=1)
            
            # Display initial measurement text if enabled
            if display_initial_text:
                # Get y-axis limits
                y_min, y_max = axs_y[j].get_ylim()
                # Place label at the center of the highlighted area
                y_init_text = axs_y[j].text(
                    initial_steps / 2, (y_min + y_max) / 2,
                    initial_text, fontsize=fontsize - 1,
                    ha='center', va='center', color='black',
                    bbox=dict(facecolor='white', edgecolor='black'))
                # Get initial text bounding box width
                init_text_width = get_text_width_in_data(
                    text_object=y_init_text, axis=axs_u[i], fig=fig)
                # Hide text box if it overflows the plot area
                if initial_steps < init_text_width:
                    y_init_text.set_visible(False)
            
            # Display Data-Driven MPC control text if enabled
            if display_control_text:
                # Get y-axis limits
                y_min, y_max = axs_y[j].get_ylim()
                # Place label at the center of the remaining area
                y_control_text = axs_y[j].text(
                    (T + initial_steps) / 2, (y_min + y_max) / 2,
                    control_text, fontsize=fontsize - 1,
                    ha='center', va='center', color='black',
                    bbox=dict(facecolor='white', edgecolor='black'))
                # Get control text bounding box width
                control_text_width = get_text_width_in_data(
                    text_object=y_control_text, axis=axs_u[0], fig=fig)
                # Hide text box if it overflows the plot area
                if (T - initial_steps) < control_text_width:
                    y_control_text.set_visible(False)
        
        axs_y[j].set_xlabel('Time step $k$', fontsize=fontsize)
        axs_y[j].set_ylabel(f'Output $y_{j+1}$', fontsize=fontsize)
        axs_y[j].legend(fontsize=fontsize)
        axs_y[j].set_xlim([0, T])
        axs_y[j].tick_params(axis='both', labelsize=fontsize)

        # Set y-limits if provided
        if y_ylimit and y_ylimit[j]:
            axs_y[j].set_ylim(y_ylimit[j])

    # Show the plot
    plt.show()

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
    there is sufficient space to prevent them from overflowing the plots.
    
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
    u_lines = []
    y_lines = []
    # Define initial measurement rectangles and texts lists
    u_rects = []
    u_rect_lines = []
    u_init_texts = []
    u_control_texts = []
    
    y_rects = []
    y_rect_lines = []
    y_init_texts = []
    y_control_texts = []

    # Define y-axis center
    u_y_axis_centers = []
    y_y_axis_centers = []
        
    # Add labels, legends and define axis limits for plots
    for i in range(m):
        # Initialize lines for inputs
        u_lines.append(axs_u[i].plot([], [], label=f'$u_{i+1}$')[0])
        # Plot input setpoint
        axs_u[i].plot(
            range(0, T), np.full(T, u_s[i, :]), label=f'$u_{i+1}^s$')
        # Set axis labels and legends
        axs_u[i].set_xlabel('Time step $k$', fontsize=fontsize)
        axs_u[i].set_ylabel(f'Input $u_{i+1}$', fontsize=fontsize)
        axs_u[i].legend(fontsize=fontsize)
        axs_u[i].tick_params(axis='both', labelsize=fontsize)

        # Define axis limits
        u_lim_min, u_lim_max = get_padded_limits(u_k[:, i], u_s[i, :])
        axs_u[i].set_xlim(0, T)
        axs_u[i].set_ylim(u_lim_min, u_lim_max)
        u_y_axis_centers.append((u_lim_min + u_lim_max) / 2)

        if initial_steps:
            # Initialize initial input rectangle
            u_rects.append(axs_u[i].axvspan(0, 0, color='gray', alpha=0.1))
            # Initialize initial input rectangle limit line
            u_rect_lines.append(axs_u[i].axvline(
                x=0, color='black', linestyle=(0, (5, 5)), linewidth=1))
            # Initialize initial input text
            u_init_texts.append(axs_u[i].text(
                initial_steps / 2, u_y_axis_centers[i],
                initial_text, fontsize=fontsize - 1, ha='center',
                va='center', color='black', bbox=dict(facecolor='white',
                                                      edgecolor='black')))
            # Initialize control input text
            u_control_texts.append(axs_u[i].text(
                (T + initial_steps) / 2, u_y_axis_centers[i],
                control_text, fontsize=fontsize - 1, ha='center',
                va='center', color='black', bbox=dict(facecolor='white',
                                                      edgecolor='black')))

    for j in range(p):
        # Initialize lines for outputs
        y_lines.append(axs_y[j].plot([], [], label=f'$y_{j+1}$')[0])
        # Plot output setpoint
        axs_y[j].plot(
            range(0, T), np.full(T, y_s[j, :]), label=f'$y_{j+1}^s$')
        # Set axis labels and legends
        axs_y[j].set_xlabel('Time step $k$', fontsize=fontsize)
        axs_y[j].set_ylabel(f'Output $y_{j+1}$', fontsize=fontsize)
        axs_y[j].legend(fontsize=fontsize, loc='lower right')
        axs_y[j].tick_params(axis='both', labelsize=fontsize)

        # Define axis limits
        y_lim_min, y_lim_max = get_padded_limits(y_k[:, j], y_s[j, :])
        axs_y[j].set_xlim(0, T)
        axs_y[j].set_ylim(y_lim_min, y_lim_max)
        y_y_axis_centers.append((y_lim_min + y_lim_max) / 2)

        if initial_steps:
            # Initialize initial output rectangle
            y_rects.append(axs_y[j].axvspan(0, 0, color='gray', alpha=0.1))
            # Initialize initial output rectangle limit line
            y_rect_lines.append(axs_y[j].axvline(
                x=0, color='black', linestyle=(0, (5, 5)), linewidth=1))
            # Initialize initial output text
            y_init_texts.append(axs_y[j].text(
                initial_steps / 2, y_y_axis_centers[j],
                initial_text, fontsize=fontsize - 1, ha='center',
                va='center', color='black', bbox=dict(facecolor='white',
                                                      edgecolor='black')))
            # Initialize control output text
            y_control_texts.append(axs_y[j].text(
                (T + initial_steps) / 2, y_y_axis_centers[j],
                control_text, fontsize=fontsize - 1, ha='center',
                va='center', color='black', bbox=dict(facecolor='white',
                                                      edgecolor='black')))
    
    # Get initial text bounding box width
    init_text_width = get_text_width_in_data(
        text_object=u_init_texts[0], axis=axs_u[0], fig=fig)
    # Get control text bounding box width
    control_text_width = get_text_width_in_data(
        text_object=u_control_texts[0], axis=axs_u[0], fig=fig)

    # Animation update function
    def update(frame):
        # Update input-output plot data
        for i in range(m):
            u_lines[i].set_data(range(0, frame+1), u_k[:frame+1, i])

            # Update initial measurements rectangle and text for input
            if initial_steps and frame <= initial_steps:
                # Update rectangle width
                u_rects[i].set_width(frame)
                # Hide initial measurement and Data-Driven MPC control texts
                u_init_texts[i].set_visible(False)
                u_control_texts[i].set_visible(False)
                # Update rectangle limit line position
                u_rect_lines[i].set_xdata([frame])
                # Show initial measurement text
                if display_initial_text and frame >= init_text_width:
                    u_init_texts[i].set_position(
                        (frame / 2, u_y_axis_centers[i]))
                    u_init_texts[i].set_visible(True)
                # Show Data-Driven MPC control text if possible
                if display_control_text and frame == initial_steps:
                    if (T - initial_steps) >= control_text_width:
                        u_control_texts[i].set_visible(True)

        for j in range(p):
            y_lines[j].set_data(range(0, frame+1), y_k[:frame+1, j])

            # Update initial measurements rectangle and text for output
            if initial_steps and frame <= initial_steps:
                # Update rectangle width
                y_rects[j].set_width(frame)
                # Hide initial measurement and Data-Driven MPC control texts
                y_init_texts[j].set_visible(False)
                y_control_texts[j].set_visible(False)
                # Update rectangle limit line position
                y_rect_lines[j].set_xdata([frame])
                # Show initial measurement text
                if display_initial_text and frame >= init_text_width:
                    y_init_texts[j].set_position(
                        (frame / 2, y_y_axis_centers[j]))
                    y_init_texts[j].set_visible(True)
                # Show Data-Driven MPC control text if possible
                if display_control_text and frame == initial_steps:
                    if (T - initial_steps) >= control_text_width:
                        y_control_texts[j].set_visible(True)

        return (u_lines + y_lines + u_rects + u_rect_lines +
                u_init_texts + u_control_texts + y_rects +
                y_init_texts + y_control_texts + y_rect_lines)

    # Create animation
    animation = FuncAnimation(fig, update, frames=T, interval=interval, blit=True)

    return animation

def save_animation(
    animation: FuncAnimation,
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
    with tqdm(total=animation.save_count, desc="Saving animation") as pbar:
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