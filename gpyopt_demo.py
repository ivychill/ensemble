"""
Resolved the following issues: 
  1. how to define a multivariate function to make it 
  amenable to optimization using GPyOpt. 
  
  2. In file bo.py, replace the existing version of function 
  def _save_model_parameter_values() with: 
  
  def _save_model_parameter_values(self):
        try:
            if self.model_parameters_iterations is None:
                self.model_parameters_iterations = self.model.get_model_parameters()
        except:
            self.model_parameters_iterations = \
            np.vstack((self.model_parameters_iterations,self.model.get_model_parameters()))
"""


import numpy as np
import GPy
import GPyOpt

def testbgo():
    def objfunc1d(x):
        return x**2

    def objfunc2d(x):
        """
        x is a 2 dimensional vector.
        """ 
        x1 = x[:, 0]
        x2 = x[:, 1]
        return x1**2 + x2**2

    maxiter = 15
    bounds1d = [{'domain':(-4, 4)}]

    bounds2d = [{'domain':(-4, 4)}, {'domain':(-4, 4)}]

    ##########
    ## 1D optimization
    optprob = GPyOpt.methods.BayesianOptimization(objfunc1d, 
        domain=bounds1d, 
        acquisition_type='EI', 
        verbosity=True)
    optprob.run_optimization(max_iter = maxiter)
    optparam = optprob.x_opt
    optfunc = optprob.fx_opt
    print "="*20
    print "1D case: "
    print "x_opt =  "+str(optparam)
    print "fx_opt = "+str(optfunc)
    print "="*20
    ########################

    
    ########################
    # 2D optimization 
    optprob = GPyOpt.methods.BayesianOptimization(objfunc2d, 
        domain=bounds2d, 
        acquisition_type='EI', 
        verbosity=True)
    optprob.run_optimization(max_iter = maxiter)
    optparam = optprob.x_opt    # <--- This is a numpy array of shape (2,)
    optfunc = optprob.fx_opt    # <--- This is a numpy array of shape (1,)
    print "="*20
    print "2D case: "
    print "x_opt =  "+str(optparam)    
    print "fx_opt = "+str(optfunc)     
    print "="*20

if __name__ == '__main__':
    func = GPyOpt.objective_examples.experiments2d.branin.f 
    #keyboard()
    testbgo()
