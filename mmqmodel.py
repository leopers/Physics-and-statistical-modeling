import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

x = np.array([20, 40, 60, 80, 20, 40, 60, 80, 20, 40, 60, 80, 20, 40, 60, 80])
y = np.array([20, 20, 20, 20, 40, 40, 40, 40, 60, 60, 60, 60, 80, 80, 80, 80])
h = np.array([20.61445294, 19.50460758, 19.30888895, 20.62554388, 19.28446684, 18.44716054, 18.53468475, 19.49063362, 19.50005064, 18.09340933, 18.5268984, 19.47455511, 20.7296302, 19.35425662, 19.5629452, 20.66162934])
sigma = 0.8
V = np.diag(sigma**2*np.ones(len(y)))
print(V)

ones_column = np.ones(len(y))  
X = np.column_stack((ones_column, np.cos(2*np.pi*x/100))) 
X = np.column_stack((X, np.cos(2*np.pi*y/100))) 
print(X)

Vi = np.linalg.inv(V)
a = np.linalg.tensorsolve(np.matmul(np.matmul(X.T,Vi), X), np.matmul(np.matmul(X.T, Vi), h))
cov_a = np.linalg.inv(np.matmul(np.matmul(X.T,Vi), X))
print(a)
print(cov_a)

res = h - np.matmul(X,a)  
S = np.matmul(res.T,res)
ngl = len(h) - 2; 
var_posteriori = S/ngl
print("sigma_posteriori = ", np.sqrt(var_posteriori))
print("var_posteriori = ", var_posteriori)

h_fit = np.matmul(X, a)
h_res = h - h_fit

chi2 = np.matmul(np.matmul(h_res.T, Vi), h_res)
p_chi2 = stats.distributions.chi2.sf(chi2, ngl)
print("Probabilidade associada à hipótese nula: ", p_chi2)
print("A hipótese nula é rejeitada se p < 5% (incerteza superestimada ou V não deveria ser diagonal pois há covariância nos dados)  ou p > 95% (incerteza subestimada ou modelo inadequado)")