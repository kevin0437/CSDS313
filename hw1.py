import numpy as np
from scipy.stats import *
import matplotlib.pyplot as plt
from tabulate import tabulate
import pandas as pd
import statsmodels.api as sm
import pylab as py


binomial = np.random.binomial(10, 0.4, 100)
bernoulli = bernoulli.rvs(0.3, size=100)
uniform = np.random.uniform(3,10,100)
normal = np.random.normal(-1,2,100)
exponential = np.random.exponential(2,100)
powerLaw = np.random.power(3,100)

mean_binomial_10 = np.mean(binomial[:10])
mean_binomial=np.mean(binomial)
mean_bernoulli_10 = np.mean(bernoulli[:10])
mean_bernoulli = np.mean(bernoulli)
mean_uniform_10 = np.mean(uniform[:10])
mean_uniform = np.mean(uniform)
mean_normal_10 = np.mean(normal[:10])
mean_normal = np.mean(normal)
mean_exponential_10 = np.mean(exponential[:10])
mean_exponential = np.mean(exponential)
mean_powerLaw_10 = np.mean(powerLaw[:10])
mean_powerLaw = np.mean(powerLaw)

var_binomial_10 = np.var(binomial[:10])
var_binomial=np.var(binomial)
var_bernoulli_10 = np.var(bernoulli[:10])
var_bernoulli = np.var(bernoulli)
var_uniform_10 = np.var(uniform[:10])
var_uniform = np.var(uniform)
var_normal_10 = np.var(normal[:10])
var_normal = np.var(normal)
var_exponential_10 = np.var(exponential[:10])
var_exponential = np.var(exponential)
var_powerLaw_10 = np.var(powerLaw[:10])
var_powerLaw = np.var(powerLaw)


mean_matrix = np.zeros([6,2])
var_matrix = np.zeros([6,2])
means = [mean_binomial_10,mean_binomial,mean_bernoulli_10,mean_bernoulli,mean_uniform_10,mean_uniform,mean_normal_10,mean_normal,mean_exponential_10,
mean_exponential,mean_powerLaw_10,mean_powerLaw]
vars = [var_binomial_10,var_binomial,var_bernoulli_10,var_bernoulli,var_uniform_10,var_uniform,var_normal_10,var_normal,var_exponential_10,
var_exponential,var_powerLaw_10,var_powerLaw]
for x in range(6):
    mean_matrix[x] = [means[2*x],means[2*x+1]]
df2 = pd.DataFrame(mean_matrix,
                   columns=['sample mean 10', 'sample mean 100'])
df2.index = ['binomial','bernoulli','uniform','normal','exponential','powerLaw']
#print(df2)
for x in range(6):
    var_matrix[x] = [vars[2*x],vars[2*x+1]]

df = pd.DataFrame(var_matrix,
                   columns=['sample var 10', 'sample var 100'])
df.index = ['binomial','bernoulli','uniform','normal','exponential','powerLaw']
#print(df)

#data = [bernoulli, binomial, uniform, normal, exponential, powerLaw]
#fig = plt.figure(figsize = (10,7))
#ax = fig.add_axes([0,0,1,1])
#bp = ax.boxplot(data,labels = ['binomial','bernoulli','uniform','normal','exponential','powerLaw'])
#plt.show()

'''
x_bernoulli = np.sort(normal)
y_bernoulli = np.arange(100)/float(100)
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('uniform CDF')
plt.plot(x_bernoulli,y_bernoulli,marker = 'o')
plt.show()
print(y_bernoulli)
from statistics import NormalDist
res = NormalDist(mu=1, sigma=0.5).inv_cdf(0.5)'''

'''plt.hist(uniform, density=True, cumulative=True, label='CDF',
         histtype='step', alpha=0.8, color='k')
plt.title('uniform CDF')
plt.show()'''

plt.hist(normal, bins = 10)
plt.title('normal')
plt.show()

'''sm.qqplot(uniform)
py.title('uniform')
py.show()'''