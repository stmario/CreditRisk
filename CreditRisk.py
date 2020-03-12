import pandas as pd
import scipy.stats as scs
import numpy as np
import csv
import math
import random

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
        if(i > 0):
            data = line[0].split(",")
            rating_to_pd[data[0]] = float(data[1])
            rating_to_threshold[data[0]] = scs.norm.ppf(float(data[1]))

# create mapping from loan id to loan specific factor loading
loanid_to_alpha = {}
loanid_to_gamma = {}
with open('Factor_Loadings.csv', "r") as f:
    reader = csv.reader(f, delimiter="\t")
    for i, line in enumerate(reader):
        if(i > 0):
            data = line[0].split(",")
            fac = -1;
            if data[1] != 0:
                fac = data[1]
            elif data[2] != 0:
                fac = data[2]
            elif data[3] != 0:
                fac = data[3]
            loanid_to_alpha[data[0]] = float(fac)
            loanid_to_gamma[data[0]] = math.sqrt(1 - pow(float(fac), 2))

# read in portfolio
# Loan, Region, Rating, Exposure, LGD
portfolio = []
with open('Portfolio.csv', "r") as f:
    reader = csv.reader(f, delimiter="\t")
    for i, line in enumerate(reader):
        l = line[0].split(",")
        if(i > 0):
            portfolio.append(l)

np.random.seed(42)  # set to fixed value for reproducability of results
n_sim = 100000  # number of Monte Carlo simulations

loss_list = []
total_loss = 0
for i in range(0,n_sim):
    print("run monte-carlo iteration " + str(i) + " of " + str(n_sim))
    samples = np.random.multivariate_normal([0,0,0],cor_matrix) # samples[0]: CH, samples[1]: EU, samples[2]: US
    for loan in portfolio:
        eps = random.uniform(0,1)
        X = loanid_to_alpha[loan[0]] * 1.0 + loanid_to_gamma[loan[0]] * eps
        if (False): # 𝑋􏰎 < 𝑇h𝑟𝑒𝑠h𝑜𝑙𝑑􏰎
            # this credit defaults
            loss = loan[4] * loan[3]
            total_loss += loss
        #else:
        #   no default, nothing to do
        pass
    # append total loss of this iteration to loss_list
    loss_list.append(total_loss)

#Step 3: Assessment of aggregated losses with loss_list

#step 4: ???

#step 5: profit.
