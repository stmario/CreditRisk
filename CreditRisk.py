import pandas as pd
import scipy.stats as scs
import numpy as np
import csv
import math
import random
from matplotlib import pyplot
import time

# read in correlation matrix and create per region correlation array
correlation = pd.read_csv('Correlation.csv')
cor_matrix = [x[1:] for x in correlation.to_numpy()]
ch_toall_cors = cor_matrix[0]
eu_toall_cors = cor_matrix[0]
us_toall_cors = cor_matrix[0]
# read in ratings and create mapping from rating to probability of default
rating_to_pd = {}
rating_to_threshold = {}
with open('PD_Table.csv', "r") as f:
    reader = csv.reader(f, delimiter="\t")
    for i, line in enumerate(reader):
        if (i > 0):
            data = line[0].split(",")
            rating_to_pd[data[0]] = float(data[1])
            rating_to_threshold[data[0]] = scs.norm.ppf(float(data[1]))

# create mapping from loan id to loan specific factor loading
loanid_to_alpha = {}
loanid_to_gamma = {}
with open('Factor_Loadings.csv', "r") as f:
    reader = csv.reader(f, delimiter="\t")
    for i, line in enumerate(reader):
        if (i > 0):
            data = line[0].split(",")
            fac = max(data[1:])
            loanid_to_alpha[data[0]] = float(fac)
            loanid_to_gamma[data[0]] = math.sqrt(1 - pow(float(fac), 2))

# read in portfolio
# Loan, Region, Rating, Exposure, LGD
portfolio = []
with open('Portfolio.csv', "r") as f:
    reader = csv.reader(f, delimiter="\t")
    for i, line in enumerate(reader):
        l = line[0].split(",")
        if (i > 0):
            portfolio.append(l)
to_float = lambda x: x[0:3] + [float(x[3])] + [float(x[4])]
portfolio = list(map(to_float, portfolio))

# num_cores = multiprocessing.cpu_count()
# print('num_cores')
# print(num_cores)

def mc_sim(samples):
    total_loss = 0
    z = {'CH': samples[0], 'EU': samples[1], 'US': samples[2]}
    count = 0
    for loan in portfolio:
        eps = np.random.normal()
        X = loanid_to_alpha[loan[0]] * z[loan[1]] + loanid_to_gamma[loan[0]] * eps
        if X < rating_to_threshold[loan[2]]:  # 𝑋􏰎 < 𝑇h𝑟𝑒𝑠h𝑜𝑙𝑑􏰎
            # this credit defaults
            loss = loan[4] * loan[3]
            total_loss += loss
            count = count + 1
        # else:
        #   no default, nothing to do
    # append total loss of this iteration to loss_list
    return total_loss


np.random.seed(42)  # set to fixed value for reproducability of results
n_sim = 100000  # number of Monte Carlo simulations
start = time.time()
loss_list = []
for i in range(0, n_sim):
    total_loss = 0
    if i % (n_sim / 100) == 0:
        print("run monte-carlo iteration ... " + str(i / (n_sim / 100)) + "%")
    samples = np.random.multivariate_normal([0, 0, 0], cor_matrix)  # samples[0]: CH, samples[1]: EU, samples[2]: US
    loss_list.append(mc_sim(samples))
end = time.time()

print('Elapsed time:')
print(end - start)
# Step 3: Assessment of aggregated losses with loss_list
expected_loss = sum(loss_list) / n_sim

fig, axs = pyplot.subplots(1, 1, sharey=True, tight_layout=True)
axs.hist(loss_list, range=(0, 6e8), bins=50)
pyplot.xscale("log")
pyplot.savefig('res')
pyplot.show()

loss_list = sorted(loss_list, reverse=True)

eads = list(map((lambda x: x[3]), portfolio))

perc5 = int(n_sim * 0.05)
perc1 = int(n_sim * 0.01)
var95 = sum(eads) - sum(loss_list[0:perc5])
var99 = sum(eads) - sum(loss_list[0:perc1])

es95 = sum(loss_list[0:perc5]) / (n_sim * 0.05)
es99 = sum(loss_list[0:perc1]) / (n_sim * 0.01)

print('expected_loss')
print(expected_loss)
print('var95')
print(var95)
print('var99')
print(var99)
print('es95')
print(es95)
print('es99')
print(es99)
# step 4: ???

# step 5: profit.
