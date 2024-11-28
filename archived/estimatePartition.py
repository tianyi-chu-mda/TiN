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

def calculate_probabilities(C_TR, C_TA, C_NR, C_NA, major, minor, totalCN, TiN):
    C_T = C_TA + C_TR
    C_N = C_NA + C_NR
    C = C_T + C_N
    C_A = C_TA + C_NA
    if C == 0: 
        return 0.0

    # Calculate likelihoods
    P_R_no_mutation = binom.pmf(C_A, n=C, p=1 - TNR)
    P_R_homozygous = binom.pmf(C_A, n=C, p=TPR)
    P_R_heterozygous = binom.pmf(C_A, n=C, p=(TPR + (1 - TNR)) / 2)
    P_R_clonal = 0.0

    # Check for totalCN = 0
    if totalCN == 0:
        P_R_clonal = binom.pmf(C_A, n=C, p=1 - TNR)
    else:
        VAF_values = np.arange(minor, major + 1) / totalCN
        for VAF in VAF_values:  
            tumor_prob = TiT * (VAF * TPR + (1 - VAF) * (1 - TNR)) + (1 - TiT) * (1 - TNR)
            normal_prob = TiN * (VAF * TPR + (1 - VAF) * (1 - TNR)) + (1 - TiN) * (1 - TNR)
            tumor_result = binom.pmf(C_TA, C_T, tumor_prob)
            normal_result = binom.pmf(C_NA, C_N, normal_prob)
            P_R_clonal += tumor_result * normal_result / len(VAF_values)

    # Combine the probabilities
    P_R = (P_R_no_mutation * p_no_mutation + 
            P_R_homozygous * p_homozygous + 
            P_R_heterozygous * p_heterozygous + 
            P_R_clonal * p_clonal)

    return -np.log(P_R)

def objective_function(TiN, data):
    total_log_likelihood = 0
    for partition in data.to_delayed():
        print("start compute", flush = True)
        partition_df = partition.compute()
        print("compute done", flush = True)
        # Convert columns to numpy arrays for numba compatibility
        tumor_alt = partition_df['tumor_alt'].to_numpy()
        tumor_ref = partition_df['tumor_ref'].to_numpy()
        normal_alt = partition_df['normal_alt'].to_numpy()
        normal_ref = partition_df['normal_ref'].to_numpy()
        minor = partition_df['minor'].to_numpy()
        major = partition_df['major'].to_numpy()
        totalCN = partition_df['totalCN'].to_numpy()
        print("End converting", flush = True)
        for i in range(len(partition_df)):
            total_log_likelihood += calculate_probabilities(
                tumor_ref[i], tumor_alt[i], normal_ref[i], normal_alt[i],
                major[i], minor[i], totalCN[i], TiN
            )
        print("Done processing this partition", flush = True)
        print(total_log_likelihood, flush = True)
    return total_log_likelihood

# Load the data as a Dask DataFrame
data = dd.read_csv("/rsrch6/home/genetics/vanloolab/secure/TCGA/LUAD_WGS_BAM/TCGA-44-2655/partitioned/combinedTables/combined_counts.txt", 
sep = "\t", header = 0, 
dtype={'chr': int, 'position': int, 'major': int, 'minor': int, 'totalCN': int, 'tumor_ref': int, 'tumor_alt': int, 'normal_ref': int, 'normal_alt': int})

print(data.head(5), flush = True)

# Run the optimization to find the best TiN
result = minimize(objective_function, x0=0, args=(data,), bounds=[(0, 1)], method='L-BFGS-B')

# Output the optimized TiN value
optimal_TiN = result.x[0]
print(f"Optimal TiN: {optimal_TiN}")