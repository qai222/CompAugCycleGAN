from skopt import load

opt = load("checkpoint.pkl")
## if loading failed try switching to a different scipy
## pip install scipy==1.7.1
## pip install scipy==1.5.3
print(opt.x)
print(opt.fun)
print(opt.func_vals)
print(opt.x_iters)
