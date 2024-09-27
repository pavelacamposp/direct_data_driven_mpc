# Define general Matplotlib line parameters for input-output plots
INPUT_LINE_PARAMS = {
    'color': 'tab:blue',
    'linestyle': 'solid',
    'linewidth': 2.0
}
OUTPUT_LINE_PARAMS = {
    'color': 'black',
    'linestyle': 'solid',
    'linewidth': 2.0
}
SETPOINT_LINE_PARAMS = {
    'color': 'red',
    'linestyle': 'solid',
    'linewidth': 2.0
}

# Define Matplotlib line parameter variations for long data sequences
INPUT_LINE_SMALL_PARAMS = INPUT_LINE_PARAMS.copy()
INPUT_LINE_SMALL_PARAMS['linewidth'] = 1.5

OUTPUT_LINE_SMALL_PARAMS = OUTPUT_LINE_PARAMS.copy()
OUTPUT_LINE_SMALL_PARAMS['linewidth'] = 1.5

SETPOINT_LINE_SMALL_PARAMS = SETPOINT_LINE_PARAMS.copy()
SETPOINT_LINE_SMALL_PARAMS['linewidth'] = 1.5

# Define general Matplotlib legend parameters
LEGEND_PARAMS = {
    'fontsize': 10,
    'handlelength': 2.7,
    'labelspacing': 0.2,
    'borderpad': 0.4,
    'fancybox': False,
    'edgecolor': 'black'
}

# Define dictionaries to group the defined parameters
INPUT_OUTPUT_PLOT_PARAMS = {
    'inputs_line_params': INPUT_LINE_PARAMS,
    'outputs_line_params': OUTPUT_LINE_PARAMS,
    'setpoints_line_params': SETPOINT_LINE_PARAMS,
    'legend_params': LEGEND_PARAMS,
}

INPUT_OUTPUT_PLOT_PARAMS_SMALL = {
    'inputs_line_params': INPUT_LINE_SMALL_PARAMS,
    'outputs_line_params': OUTPUT_LINE_SMALL_PARAMS,
    'setpoints_line_params': SETPOINT_LINE_SMALL_PARAMS,
    'legend_params': LEGEND_PARAMS,
}