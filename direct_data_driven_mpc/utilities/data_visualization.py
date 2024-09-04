from typing import Tuple, Optional, List

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
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 300,
    u_ylimit: Optional[List[Tuple[float, float]]] = None,
    y_ylimit: Optional[List[Tuple[float, float]]] = None,
    fontsize: int = 12
) -> None:
    """
    Plot input-output data with setpoints in a Matplotlib figure.

    This function creates a 2-row subplot, with the first row containing
    control inputs, and the second row, system outputs. Each subplot contains
    the data series for each data sequence and its corresponding setpoint as
    constant lines across the series.

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
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 300,
    interval: int = 10,
    fontsize: int = 12
) -> FuncAnimation:
    """
    Create a Matplotlib animation showing the progression of input-output data
    over time.
    
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

    # Initialize lines for inputs and outputs
    u_lines = [axs_u[i].plot([], [], label=f'$u_{i+1}$')[0]
               for i in range(m)]
    y_lines = [axs_y[j].plot([], [], label=f'$y_{j+1}$')[0]
               for j in range(p)]
    
    # Plot setpoint lines
    for i in range(m):
        # Plot input setpoint
        axs_u[i].plot(
            range(0, T), np.full(T, u_s[i, :]), label=f'$u_{i+1}^s$')
    for j in range(p):
        # Plot output setpoint
        axs_y[j].plot(
            range(0, T), np.full(T, y_s[j, :]), label=f'$y_{j+1}^s$')
    
    # Add labels, legends and define axis limits for plots
    for i in range(m):
        # Set axis labels and legends
        axs_u[i].set_xlabel('Time step $k$', fontsize=fontsize)
        axs_u[i].set_ylabel(f'Input $u_{i+1}$', fontsize=fontsize)
        axs_u[i].legend(fontsize=fontsize)
        axs_u[i].tick_params(axis='both', labelsize=fontsize)

        # Define axis limits
        u_lim_min, u_lim_max = get_padded_limits(u_k[:, i], u_s[i, :])
        axs_u[i].set_xlim(0, T)
        axs_u[i].set_ylim(u_lim_min, u_lim_max)
    for j in range(p):
        # Set axis labels and legends
        axs_y[j].set_xlabel('Time step $k$', fontsize=fontsize)
        axs_y[j].set_ylabel(f'Output $y_{j+1}$', fontsize=fontsize)
        axs_y[j].legend(fontsize=fontsize, loc='lower right')
        axs_y[j].tick_params(axis='both', labelsize=fontsize)

        # Define axis limits
        y_lim_min, y_lim_max = get_padded_limits(y_k[:, j], y_s[j, :])
        axs_y[j].set_xlim(0, T)
        axs_y[j].set_ylim(y_lim_min, y_lim_max)

    # Animation update function
    def update(frame):
        # Update input-output plot data
        for i in range(m):
            u_lines[i].set_data(range(0, frame+1), u_k[:frame+1, i])
        for j in range(p):
            y_lines[j].set_data(range(0, frame+1), y_k[:frame+1, j])

        return u_lines + y_lines

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