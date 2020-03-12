import pandas as pd
import sklearn as skl
import numpy as np
import csv
import math

# read in correlation matrix and create per region correlation array
correlation = pd.read_csv('Correlation.csv')
cor_matrix = [x[1:] for x in correlation.to_numpy()]
ch_toall_cors = cor_matrix[0]
eu_toall_cors = cor_matrix[0]
us_toall_cors = cor_matrix[0]
# read in ratings and create mapping from rating to probability of default
rating_to_pd = {}
with open('PD_Table.csv', "r") as f:
    reader = csv.reader(f, delimiter="\t")
    for i, line in enumerate(reader):
        if(i > 0):
            data = line[0].split(",")
            rating_to_pd[data[0]] = float(data[1])

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
portfolio = []
with open('Portfolio.csv', "r") as f:
    reader = csv.reader(f, delimiter="\t")
    for i, line in enumerate(reader):
        l = line[0].split(",")
        portfolio.append(l)
for i in portfolio:
    print(i)

np.random.seed(42)  # set to fixed value for reproducability of results
n_sim = 100000  # number of Monte Carlo simulations

loss_list = []
total_loss = 0
for i in range(0,n_sim):
    samples = np.random.multivariate_normal([0,0,0],cor_matrix) # samples[0]: CH, samples[1]: EU, samples[2]: US
    for loan in portfolio:
        #flip coin in [0,1]. this is idiosyncratic risk factor realisation?
        #Calculate the loan-level Credit Quality Processes: 𝑋􏰎 = 𝛼􏰏􏰐􏰑􏰎􏰆􏰈,􏰎 ⋅ 𝑍􏰏􏰐􏰑􏰎􏰆􏰈 + 𝛾􏰎 ⋅ 𝜖􏰎
        #if (𝑋􏰎 < 𝑇h𝑟𝑒𝑠h𝑜𝑙𝑑􏰎)
        #   this credit defaults
        #   calculate loss with EAD*LGD
        #   total_loss += calculated loss
        #else:
        #   no default, nothing to do
        pass
    loss_list.append(total_loss)

#Step 3: Assessment of aggregated losses with loss_list

#step 4: ???

#step 5: profit.
