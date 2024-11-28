import numpy as np
import pandas as pd
import dask.dataframe as dd
from scipy.stats import binom
from scipy.optimize import minimize

TPR = 0.9  # alternate and seq'd as alternate
TNR = 0.95  # reference and seq'd as reference
TiT = 0.27

# prior probabilities of four conditions
p_homozygous = 0.002   # prior probability of homozygous germline mutation
p_heterozygous = 0.002 # prior probability of heterozygous germline mutation
p_clonal = 10e-6       # prior probability of clonal mutation
p_no_mutation = 1 - p_homozygous - p_heterozygous - p_clonal # add up to 1

def calculate_probabilities(row, TiN):
    C_TA = row['tumor_alt']
    C_TR = row['tumor_ref']
    C_NA = row['normal_alt']
    C_NR = row['normal_ref']
    C = C_TA + C_TR + C_NA + C_NR
    C_A = C_TA + C_NA
    if C == 0: return 0

    minor = row['minor']
    major = row['major']
    totalCN = row['totalCN']
    mult = list(range(minor, major + 1))
    vaf = [x/(totalCN) for x in mult]

    # Calculate likelihoods
    P_R_no_mutation = binom.pmf(C_A, n=C, p=1 - TNR)
    P_R_homozygous = binom.pmf(C_A, n=C, p=TPR)
    P_R_heterozygous = binom.pmf(C_A, n=C, p=(TPR + (1 - TNR)) / 2)
    P_R_clonal = 0
    for VAF in vaf:
        P_R_clonal += (
            binom.pmf(
                C_TA, n=C_TA + C_TR, p=TiT * (VAF * TPR + (1 - VAF) * (1 - TNR)) + (1 - TiT) * (1 - TNR)
                ) *
            binom.pmf(
                C_NA, n=C_NA + C_NR, p=TiN * (VAF * TPR + (1 - VAF) * (1 - TNR)) + (1 - TiN) * (1 - TNR)
                )
        )/len(vaf)

    # Combine the probabilities
    P_R = (P_R_no_mutation * p_no_mutation + 
            P_R_homozygous * p_homozygous + 
            P_R_heterozygous * p_heterozygous + 
            P_R_clonal * p_clonal)

    return -np.log(P_R)

def objective_function(TiN, data):
    total_log_likelihood = 0
    for partition in data.to_delayed():  # Process each chunk independently
        partition_df = partition.compute()
        total_log_likelihood += sum(calculate_probabilities(row, TiN) for _, row in partition_df.iterrows())
    return total_log_likelihood

def callback(xk):
    print(f"Current TiN estimate: {xk[0]}")
    with open("optimization_progress.log", "a") as log_file:
        log_file.write(f"Current TiN estimate: {xk[0]}\n")

# Load the data as a Dask DataFrame
data = dd.read_csv("/rsrch6/home/genetics/vanloolab/secure/TCGA/LUAD_WGS_BAM/TCGA-44-2655/partitioned/combinedTables/combined_chr.txt", sep = "\t", header = 0, 
dtype={'chr': int, 'position': int, 'minor': int, 'major': int, 'totalCN': int, 'normal_ref': int, 'normal_alt': int, 'tumor_ref': int, 'tumor_alt': int})

print(data.head(5), flush = True)

# Run the optimization to find the best TiN
result = minimize(objective_function, x0=0, args=(data,), bounds=[(0, 1)], method='L-BFGS-B')

# Output the optimized TiN value
optimal_TiN = result.x[0]