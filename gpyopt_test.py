
# %pylab inline
import GPy
import GPyOpt
from numpy.random import seed
import matplotlib

seed(123)

f_true = GPyOpt.objective_examples.experiments2d.sixhumpcamel()
f_sim = GPyOpt.objective_examples.experiments2d.sixhumpcamel(sd = 0.1)
bounds =[{'name': 'var_1', 'type': 'continuous', 'domain': f_true.bounds[0]},
         {'name': 'var_2', 'type': 'continuous', 'domain': f_true.bounds[1]}]

print('plot f_true')
f_true.plot()

# Creates three identical objects that we will later use to compare the optimization strategies
myBopt2D = GPyOpt.methods.BayesianOptimization(f_sim.f,
                                              domain=bounds,
                                              model_type = 'GP',
                                              acquisition_type='EI',
                                              normalize_Y = True,
                                              acquisition_weight = 2)

# runs the optimization for the three methods
max_iter = 40  # maximum time 40 iterations
max_time = 60  # maximum time 60 seconds

myBopt2D.run_optimization(max_iter,max_time,verbosity=False)

print('plot_acquisition')
myBopt2D.plot_acquisition()
print('plot_convergence')
myBopt2D.plot_convergence()