import numpy as np
import pandas as pd
from numba import njit
from numpy.random import dirichlet
import sys

sample = sys.argv[1]
ID = sys.argv[2]
path = sys.argv[3]
method = sys.argv[4]

TPR = 0.9  # alternate and seq'd as alternate
TNR = 0.95  # reference and seq'd as reference
if sample == "TCGA-44-2655":
    TiT = 0.27
elif sample == "TCGA-50-6592":
    TiT = 0.22
elif sample == "TCGA-55-6986":
    TiT = 0.40

@njit
def generate_dirichlet(alpha_prior):
    gamma_samples = np.empty(len(alpha_prior))
    for i in range(len(alpha_prior)):
        gamma_samples[i] = np.random.gamma(alpha_prior[i], 1.0)  # Sample from Gamma distribution
    
    # Normalize to produce Dirichlet distribution
    sum_gamma = np.sum(gamma_samples)
    return gamma_samples / sum_gamma

@njit
def binomial_pmf(k, n, p):
    # Calculate binomial coefficient nCk = n! / (k!(n-k)!)
    if k > n or k < 0:
        return 0.0
    coeff = 1.0
    for i in range(k):
        coeff *= (n - i) / (i + 1)
    return coeff * (p ** k) * ((1 - p) ** (n - k))

@njit
def calculate_probabilities(C_TR, C_TA, C_NR, C_NA, major, minor, totalCN, TiN, AF, alpha_prior):
    C_T = C_TA + C_TR
    C_N = C_NA + C_NR
    C = C_T + C_N
    C_A = C_TA + C_NA
    if C == 0: 
        return 0.0, alpha_prior

    if totalCN == 0:
        x_N = 0
        y_N = 1
        x_T = 0
        y_T = 1
    else:
        x_N = TiN * totalCN / ((1 - TiN) * 2 + TiN * totalCN)
        y_N = 1 - x_N
        x_T = TiT * totalCN / ((1 - TiT) * 2 + TiT * totalCN)
        y_T = 1 - x_T
    
    # Calculate likelihoods
    # No mutation:
    P_TA_N = binomial_pmf(C_TA, n=C_T, p=1-TNR)
    P_NA_N = binomial_pmf(C_NA, n=C_N, p=1-TNR)
    P_R_no_mutation = P_TA_N * P_NA_N

    # Homozygous germline:
    P_TA_O = binomial_pmf(C_TA, n=C_T, p=TPR)
    P_NA_O = binomial_pmf(C_NA, n=C_N, p=TPR)
    P_R_homozygous = P_TA_O * P_NA_O

    if totalCN == 0:
        P_TA_E = binomial_pmf(C_TA, n=C_T, p=y_T*(TPR+1-TNR)/2)
        P_NA_E = binomial_pmf(C_NA, n=C_N, p=y_N*(TPR+1-TNR)/2)
        P_R_heterozygous = P_TA_E * P_NA_E

        P_TA_C = binomial_pmf(C_TA, n=C_T, p=y_T*(1-TNR))
        P_NA_C = binomial_pmf(C_NA, n=C_N, p=y_N*(1-TNR))
        P_R_clonal = P_TA_C * P_NA_C

    else:
        # Heterozygous germline:
        p_T_minor = TPR*(y_T/2+x_T*minor/totalCN)+(1-TNR)*(y_T/2+x_T*major/totalCN)
        p_T_major = TPR*(y_T/2+x_T*major/totalCN)+(1-TNR)*(y_T/2+x_T*minor/totalCN)
        p_N_minor = TPR*(y_N/2+x_N*minor/totalCN)+(1-TNR)*(y_N/2+x_N*major/totalCN)
        p_N_major = TPR*(y_N/2+x_N*major/totalCN)+(1-TNR)*(y_N/2+x_N*minor/totalCN)
        P_TA_E = binomial_pmf(C_TA, n=C_T, p=p_T_minor)/2 + binomial_pmf(C_TA, n=C_T, p=p_T_major)/2
        P_NA_E = binomial_pmf(C_NA, n=C_N, p=p_N_minor)/2 + binomial_pmf(C_NA, n=C_N, p=p_N_major)/2
        P_R_heterozygous = P_TA_E * P_NA_E

        # Clonal: 
        P_TA_C, P_NA_C = 0, 0
        multiplicity = np.arange(1, major + 1)
        for i in multiplicity:
            p_TA_C = x_T*(TPR*i/totalCN+(1-TNR)*(totalCN-i)/totalCN)+y_T*(1-TNR)
            p_NA_C = x_N*(TPR*i/totalCN+(1-TNR)*(totalCN-i)/totalCN)+y_N*(1-TNR)
            P_TA_C += binomial_pmf(C_TA, n=C_T, p=p_TA_C)/major
            P_NA_C += binomial_pmf(C_NA, n=C_N, p=p_NA_C)/major
        P_R_clonal = P_TA_C * P_NA_C

    # Combine the probabilities
    p_no_mutation, p_homozygous, p_heterozygous, p_clonal = generate_dirichlet(alpha_prior)

    #p_clonal = alpha / (alpha + beta)
    #p_germline = 1 - p_clonal
    #p_no_mutation = (1-AF)*(1-AF)*p_germline
    #p_homozygous = AF*AF*p_germline
    #p_heterozygous = 2*(1-AF)*AF*p_germline

    P_no_mutation = P_R_no_mutation * p_no_mutation
    P_homozygous = P_R_homozygous * p_homozygous
    P_heterozygous = P_R_heterozygous * p_heterozygous
    P_clonal = P_R_clonal * p_clonal
    P_R = (P_no_mutation + P_homozygous + P_heterozygous + P_clonal)

    # Update priors
    # determine to which mutation scenario this row belongs:
    posteriors = [P_no_mutation, P_homozygous, P_heterozygous, P_clonal]
    best_scenario = np.argmax(np.array(posteriors))  # Returns the index of the highest probability
    # update with observations
    #if best_scenario == 3:
        #alpha += 1
    #else:
        #beta += 1

    success = [0, 0, 0, 0]
    success[best_scenario] += 1
    alpha_prior = [a + b for a, b in zip(alpha_prior, success)]

    #print(P_no_mutation/P_R, P_homozygous/P_R, P_heterozygous/P_R, P_clonal/P_R)
    #print(best_scenario)
    #print(alpha_prior)
    #print(P_R_no_mutation, P_R_homozygous, P_R_heterozygous, P_R_clonal)
    #print(P_R)
    #return -np.log(P_R), alpha_prior
    return -np.log(P_R), alpha_prior

@njit
def objective(TiN_grid, data):
    best_TiN = 0
    min_neg_log_likelihood = np.inf
    for TiN in TiN_grid:
        negative_log_likelihood = 0
        alpha_prior = [1, 1, 1, 1]
        #alpha, beta = 1, 1
        for i in range(data_array.shape[0]):
            row = data_array[i]
            likelihood, alpha_prior = calculate_probabilities(
            #likelihood, alpha, beta = calculate_probabilities(
                row[0], row[1], row[2], row[3],
                row[4], row[5], row[6], TiN, row[7], alpha_prior
                #row[4], row[5], row[6], TiN, row[7], alpha, beta
            )
            negative_log_likelihood += likelihood
        if negative_log_likelihood < min_neg_log_likelihood:
            min_neg_log_likelihood = negative_log_likelihood
            best_TiN = TiN
    return best_TiN, min_neg_log_likelihood

TiN_grid = np.linspace(0, 1, 100)

data = pd.read_csv(
    #"/rsrch6/home/genetics/vanloolab/secure/TCGA/LUAD_WGS_BAM/TCGA-44-2655/partitioned/AFcombined/filtered_final.txt",
    f"/rsrch8/home/genetics/tchu/TCGA_LUAD/step6_estimate/subsamples/{sample}/random_sample{ID}.txt", 
    sep = "\t", header = 0, 
dtype={'chr': int, 'position': int, 'major': int, 'minor': int, 'totalCN': int, 'tumor_ref': int, 'tumor_alt': int, 'normal_ref': int, 'normal_alt': int, 'AF': float},
index_col=False)
data_array = data[['tumor_ref', 'tumor_alt', 'normal_ref', 'normal_alt', 'major', 'minor', 'totalCN', 'AF']].to_numpy()

optimal_TiN, min_neg_log_likelihood = objective(TiN_grid, data_array)

with open(f'{path}/Dirichlet_estimated_outputs_{method}_{ID}.txt', 'w') as file:
    file.write(f"Optimal TiN: {optimal_TiN}\n")
    file.write(f"Minimum Negative Log Likelihood: {min_neg_log_likelihood}\n")
print("Optimal TiN is estimated as: ", optimal_TiN)
