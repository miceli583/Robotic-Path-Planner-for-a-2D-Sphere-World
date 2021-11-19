import math
import numbers
import numpy as np
from math import nan
from matplotlib import cm, pyplot as plt
from scipy.linalg import block_diag


def numel(var):
    if isinstance(var, numbers.Number):
        size = int(1)
    elif isinstance(var, np.ndarray):
        size = var.size
    else:
        breakpoint()
        raise NotImplementedError(f'number of elements for type {type(var)}')
    return size

class Grid:
    def __init__(self, xx_grid, yy_grid):
        self.xx_grid = xx_grid
        self.yy_grid = yy_grid

    def eval(self, fun):
        dim_domain = [numel(self.xx_grid), numel(self.yy_grid)]
        dim_range = [numel(fun(np.array([[0], [0]])))]
        fun_eval = np.nan * np.ones(dim_domain + dim_range)
        for idx_x in range(0, dim_domain[0]):
            for idx_y in range(0, dim_domain[1]):
                x_eval = np.array([[self.xx_grid[idx_x]],
                                   [self.yy_grid[idx_y]]])
                fun_eval[idx_x, idx_y, :] = np.reshape(fun(x_eval),
                                                       [1, 1, dim_range[0]])

        # If the last dimension is a singleton, remove it
        if dim_range == [1]:
            fun_eval = np.reshape(fun_eval, dim_domain)

        return fun_eval

    def mesh(self):
        return np.meshgrid(self.xx_grid, self.yy_grid)


def clip(val, threshold):
    if isinstance(val, np.ndarray):
        val_norm = np.linalg.norm(val)
        if val_norm > threshold:
            val /= val_norm
            #val = val*threshold/val_norm
    elif isinstance(val, numbers.Number):
        if val > threshold:
            val = threshold
        if np.isnan(val):
            val = threshold
    else:
        raise ValueError('Numeric format not recognized')
    return val


def field_plot_threshold(f_handle, threshold=10, nb_grid=61):
    xx_grid = np.linspace(-11, 11, nb_grid)
    yy_grid = np.linspace(-11, 11, nb_grid)
    grid = Grid(xx_grid, yy_grid)

    f_handle_clip = lambda val: clip(f_handle(val), threshold)
    f_eval = grid.eval(f_handle_clip)

    [xx_mesh, yy_mesh] = grid.mesh()
    f_dim = numel(f_handle_clip(np.zeros((2, 1))))
    if f_dim == 1:
        # scalar field
        fig = plt.gcf()
        axis = fig.add_subplot(111, projection='3d')

        axis.plot_surface(xx_mesh,
                          yy_mesh,
                          f_eval.transpose(),
                          cmap=cm.gnuplot2)
        axis.view_init(90, -90)
    elif f_dim == 2:
        # vector field
        # grid.eval gives the result transposed with respect to what meshgrid expects
        f_eval = f_eval.transpose((1, 0, 2))
        # vector field
        plt.quiver(xx_mesh,
                   yy_mesh,
                   f_eval[:, :, 0],
                   f_eval[:, :, 1],
                   angles='xy',
                   scale_units='xy')
    else:
        raise NotImplementedError(
            'Field plotting for dimension greater than two not implemented')

    plt.xlabel('x')
    plt.ylabel('y')


class Sphere:
    def __init__(self, center, radius, distance_influence):
        self.center = center
        self.radius = radius
        self.distance_influence = distance_influence

    def plot(self, color, ax = None):
        # Add circle as a patch
        if self.radius > 0:
            # Circle is filled in
            kwargs = {'facecolor': (0.3, 0.3, 0.3)}
        else:
            # Circle is hollow
            kwargs = {'fill': False}

        center = (self.center[0, 0], self.center[1, 0])
        ax.add_patch(
            plt.Circle(center,
                       radius=abs(self.radius),
                       edgecolor=color,
                       **kwargs))
        return ax

    def distance(self, points):
        d_points_sphere = np.linalg.norm(points-self.center,axis = 0)-abs(self.radius)
        if(self.radius<0):
            d_points_sphere *= -1
        
        return d_points_sphere

    def distance_grad(self, points):
        #ADD CODE TO HANDLE SINGULARITY
        grad_d_points_sphere = (points-self.center)/np.linalg.norm(points-self.center,axis = 0)
        if(self.radius<0):
            grad_d_points_sphere *= -1
        return grad_d_points_sphere

    def beta(self, points):
        beta_val = (np.linalg.norm(points-self.center,axis = 0)**2) - (abs(self.radius)**2)
        if(self.radius<0):
            beta_val *= -1
        return beta_val

    def beta_grad(self, points):
        beta_grad = 2*(points-self.center)
        if(self.radius<0):
            beta_grad *= -1
        return beta_grad
