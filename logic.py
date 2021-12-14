import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=5)


def solve(A, y, mult=False, max_log=3):
    
    if mult:
        y = np.log(1 + (max_log - 1) * y)
        A = np.log((1 + max_log) / 2 + (max_log - 1) * A / 2)
        
    return np.linalg.lstsq(A, y)[0]


def forward(A, coeff, mult=False, max_log=3):
    
    if mult:
        A_log = np.log((1 + max_log) / 2 + (max_log - 1) * A / 2)
        y_log = A_log @ coeff.T
        y = (np.exp(y_log) - 1) / (max_log - 1)
        return y
        
    else:
        return A @ coeff.T

def get_subscript(symbol):
    if symbol == 'i': return u'\u1d62'
    return chr(ord(u'\u2080') + symbol)


def split_data_by_features(x, feature_lengths):
    return [x[:, feature_range[0]: feature_range[1]] for feature_range in get_ranges(feature_lengths)]


def read_input_from_files(feature_filenames, labels_filename, rows_limit=None):
    
    feature_lengths = ()
    x = []
    
    for feature_filename in feature_filenames:
        x_var = np.loadtxt(feature_filename)[:, 1:]
        feature_lengths += (x_var.shape[-1], )
        x.append(x_var)
    
    x = np.concatenate(x, axis=-1)
    
    y = np.loadtxt(labels_filename)[:, 1:]
        
    if rows_limit and (x.shape[0] > rows_limit):
        x = x[:rows_limit]
        y = y[:rows_limit]
        
    return (x, feature_lengths), y 


def normalize_data(x, min_max_data=False):
    min_values = np.min(x, axis=0)
    max_values = np.max(x, axis=0)
    
    divide = np.array(max_values - min_values)
    divide[divide < 1e-5] = 1
    
    x = (x - min_values) / divide
    if min_max_data:
        return x, (min_values, max_values)
    return x


def transform_data(x, min_values, max_values):
    divide = np.array(max_values - min_values)
    divide[divide < 1e-5] = 1
    
    x = (x - min_values) / divide
    return x


def denormalize_data(x, norm_values):
    min_values, max_values = norm_values
    x = x * (max_values - min_values) + min_values
    return x


def polynomial(polynomial_type):
    
    def сhebyshev_first(n, x): 
        return np.cos(n * np.arccos(x))
    
    def chebyshev_second(n, x):
        
        if x == 1:
            if n == 0: return 1
            if n == 1: return 2
            if n == 2: return 3
        
        acos = np.arccos(x)
        return np.sin((n + 1) * acos) / acos
    
    def legendre(n, x):
        
        if n == 0:
            return 0.5
        if n == 1:
            return x
        
        return (2 * n + 1) / (n + 1) * x * legendre(n - 1, x) - n / (n + 1) * legendre(n - 2, x)
        
    def hermite(n, x):
        
        if n == 0:
            return 1
        if n == 1:
            return x
            
        return x * hermite(n - 1, x) -(n - 1) * hermite(n - 2, x)
    
    def laguerre(n, x):
        
        if n == 0:
            return 0.5
        if n == 1:
            return - x + 1
    
        return ((2 * n - x - 1) * laguerre(n - 1, x) - (n - 1) * laguerre(n - 2, x)) / n
    
    # return function with name corresponding to polynome_type variable, 
    # second parameter is a default value
    return locals().get(polynomial_type, сhebyshev_first)


def timeseries_to_fixed_window_array(timeseries, window_size=10):
   
    output_shape = (timeseries.shape[0] - window_size, window_size) + timeseries.shape[1:]
    fixed_window_array = np.empty(shape=output_shape, dtype=timeseries.dtype)
    for ind in range(output_shape[0]):
        fixed_window_array[ind] = timeseries[ind:ind + window_size]
    
    return fixed_window_array

def timeseries_to_fixed_window_array_padded(timeseries, window_size=10):
    padding = np.ones(shape=(window_size, ) + timeseries.shape[1:]) * timeseries[0]
    padded_timeseries = np.concatenate([padding, timeseries])
    return timeseries_to_fixed_window_array(padded_timeseries, window_size=window_size)
    

def get_ranges(lengths):
    
    ranges = []
    ind = 0
    for length in lengths:
        ranges.append((ind, ind + length))
        ind += length
        
    return ranges


def create_equation_matrix(x, polynomial_type='сhebyshev_first', polynomial_degree=2):
    
    polynomial_function = polynomial(polynomial_type)
        
    A = np.empty((x.shape[0], x.shape[1] * polynomial_degree), dtype=np.float32)
    
    for row_ind in range(x.shape[0]):
                
        for variable_ind in range(x.shape[1]):
            variable_value = 2 * x[row_ind, variable_ind] - 1
            for degree_ind in range(polynomial_degree):
                A[row_ind, polynomial_degree * variable_ind + degree_ind] = polynomial_function(degree_ind, variable_value)
    
    return A

def concat_equation_matrices(matrices):
    return np.concatenate(matrices, axis=1)

def save_graph(plot_vars, var_names, filename='graph.png', y_lim=None, colors=('p', 'b', 'g'), crit=None, warn=None):
    plt.figure(figsize=(10,6))
    
    axes = plt.gca()
    if y_lim is not None:
        axes.set_ylim(y_lim)
    
    n = np.arange(len(plot_vars[0])) * 10
    
    for plot_var, color in zip(plot_vars, colors):
        plt.plot(n, plot_var, color)
        
    if crit is not None:
        plt.plot(n, [crit] * n.shape[0], 'r')
        var_names = var_names + ['Критичний рівень']
        
    if warn is not None:
        plt.plot(n, [warn] * n.shape[0], 'y')
        var_names = var_names + ['Загроза нестабільного стану']
        
    plt.legend(var_names)
    plt.savefig(filename, bbox_inches = 'tight')


def save_graph_sequence(
    plot_vars, var_names, filenames='graph_{0}.png', step=10, y_lim=None, colors=('p', 'b', 'g'), crit=None, warn=None):
    
    for end_point in range(0, plot_vars[0].shape[0] - 1, step):
        save_graph(
            [plot_var[:end_point] for plot_var in plot_vars], 
            var_names=var_names,
            filename=filenames.format(end_point // step + 1), 
            y_lim=y_lim, colors=colors, crit=crit, warn=warn)
        
    graph_amount = (plot_vars[0].shape[0] - 1) // step + 1
    save_graph(plot_vars, var_names, filename=filenames.format(graph_amount), y_lim=y_lim, colors=colors, crit=crit, warn=warn)
    return graph_amount


def calculate_trustworthness(y, approx):
    return np.max(np.abs(y - approx)) / (max(y.max(), approx.max()) - min(y.min(), approx.min()))


def calculate_risk(y, return_max_risk=False):
    window_size = 8
    
    risks = np.empty(3, dtype=np.float32)
    risks[0] = 0.2 * (1 - y[1][-1] / y[1].max()) + (1 - 0.2) * np.abs((y[0][:1] - y[0][1:])[-window_size:]).max()
    risks[1] = 1 - y[1][-1] / y[1].max()
    risks[2] = np.abs((y[2][:1] - y[2][1:])[-window_size:]).max()
    
    risks[1] = risks[1] ** 3
    
    risk = (1 - np.prod(1 - risks)) ** 0.8
    if return_max_risk: return (risk, np.argmax(risks))
    return  risk

def exp_average(x, order=5):
    coeffs = np.array([np.exp(ind + 1) for ind in range(-order, 0)])
    coeffs = coeffs / coeffs.sum()
    
    size = len(x)
    x = np.array([x[0]] * (order - 1) + list(x))
    res = np.empty(size)
    for ind in range(size):
        res[ind] = np.dot(x[ind: ind + order], coeffs)
    return np.array(res)