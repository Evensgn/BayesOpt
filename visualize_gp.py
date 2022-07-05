from sklearn.gaussian_process.kernels import Matern, RBF
import numpy as np
import matplotlib.pyplot as plt
from main import generate_samples_of_target_func
from mpl_toolkits.mplot3d import Axes3D


if __name__ == "__main__":
    n_dim = 2
    inner_gp_nu = 2.5
    inner_gp_length_scale = 1.0
    inner_gp_lambda = 1.0
    n_test_funcs = 5
    n_grid_each_dim = 20

    print('set parameter values')

    inner_gp_params = {
        'kernel': inner_gp_lambda * Matern(nu=inner_gp_nu, length_scale=inner_gp_length_scale),
        'alpha': 1e-6,
        'normalize_y': True,
        'n_restarts_optimizer': 5,
        'random_state': None
    }

    # generate grid points
    grid_dims = []
    bounds = np.array([[-1.0, 1.0]] * n_dim)
    for i in range(len(bounds)):
        grid_dims.append(np.linspace(bounds[i][0], bounds[i][1], n_grid_each_dim))
    mesh_dims = np.meshgrid(*grid_dims)
    grid_points = np.vstack(list(map(np.ravel, mesh_dims))).T
    B, D = np.meshgrid(grid_dims[0], grid_dims[1])

    print('generated grid points')

    # generate test functions
    test_funcs, test_func_max_values = generate_samples_of_target_func(
        inner_gp_params, grid_points, n_test_funcs, None
    )
    test_func_bounds = [bounds] * n_test_funcs

    print('generated test functions')

    for i in range(n_test_funcs):
        func_values = test_funcs[i].func_values().reshape(B.shape)
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(B, D, func_values)
        plt.show()





