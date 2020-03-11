import pandas as pd
import sklearn as skl
import numpy as np

correlation = pd.read_csv('Correlation.csv')
print(correlation)
cor_matrix = [x[1:] for x in correlation.to_numpy()]
print(cor_matrix)
factor_loadings = pd.read_csv('Factor_Loadings.csv')
pd_table = pd.read_csv('PD_Table.csv')
portfolio = pd.read_csv('Portfolio.csv')

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
    loss_list.append(total_loss)

#Step 3: Assessment of aggregated losses with loss_list

#step 4: ???

#step 5: profit.
