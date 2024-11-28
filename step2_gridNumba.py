import numpy as np
import pandas as pd
from numba import njit
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
def logaddexp_reduce(values):
    """ Manually implement logaddexp.reduce for Numba """
    result = values[0]
    for i in range(1, len(values)):
        result = np.logaddexp(result, values[i])
    return result

@njit
def log_factorial(n):
    """
    Compute the logarithm of n! using Stirling's approximation.
    """
    if n == 0 or n == 1:
        return 0.0
    return n * np.log(n) - n + 0.5 * np.log(2 * np.pi * n)

@njit
def log_n_choose_k(n, k):
    """
    Compute the natural logarithm of the binomial coefficient C(n, k).
    log(C(n, k)) = log(n!) - log(k!) - log((n-k)!)
    Uses Stirling's approximation for large factorials.
    """
    if k > n or k < 0:
        return -np.inf  # Invalid cases
    return log_factorial(n) - log_factorial(k) - log_factorial(n - k)

@njit
def log_binomial_pmf(k, n, p):
    """
    Compute the logarithm of the binomial probability mass function.
    """
    if k < 0 or k > n or p <= 0.0 or p >= 1.0:
        return -np.inf  # Invalid inputs lead to log(0)

    # Use log_n_choose_k for the binomial coefficient
    log_nck = log_n_choose_k(n, k)
    log_prob = log_nck + k * np.log(p) + (n - k) * np.log(1 - p)
    return log_prob

@njit
def calculate_probabilities(C_TR, C_TA, C_NR, C_NA, major, minor, totalCN, AF, TiN):
    C_T = C_TA + C_TR
    C_N = C_NA + C_NR
    C = C_T + C_N
    C_A = C_TA + C_NA
    if C == 0: 
        return 0.0

    if AF <= 0.0001:
        AF = 0.0001
    elif AF >= 0.9999:
        AF = 0.9999

    x_T = TiT * totalCN / ((1 - TiT) * 2 + TiT * totalCN)
    y_T = 1 - x_T
    if totalCN == 0:
        x_N = 0
        y_N = 1
    else:
        x_N = TiN * totalCN / ((1 - TiN) * 2 + TiN * totalCN)
        y_N = 1 - x_N

    # Calculate likelihoods
    # No mutation:
    P_TA_N = log_binomial_pmf(C_TA, n=C_T, p=1-TNR)
    P_NA_N = log_binomial_pmf(C_NA, n=C_N, p=1-TNR)
    P_R_no_mutation = P_TA_N + P_NA_N

    # Homozygous germline:
    P_TA_O = log_binomial_pmf(C_TA, n=C_T, p=TPR)
    P_NA_O = log_binomial_pmf(C_NA, n=C_N, p=TPR)
    P_R_homozygous = P_TA_O + P_NA_O

    if totalCN == 0:
        P_TA_E = log_binomial_pmf(C_TA, n=C_T, p=(TPR+1-TNR)/2)
        P_NA_E = log_binomial_pmf(C_NA, n=C_N, p=(TPR+1-TNR)/2)
        P_R_heterozygous = P_TA_E + P_NA_E

        P_TA_C = log_binomial_pmf(C_TA, n=C_T, p=(1-TNR))
        P_NA_C = log_binomial_pmf(C_NA, n=C_N, p=(1-TNR))
        P_R_clonal = P_TA_C + P_NA_C
    else:
        # Heterozygous germline:
        p_T_minor = TPR*(y_T/2+x_T*minor/totalCN)+(1-TNR)*(y_T/2+x_T*major/totalCN)
        p_T_major = TPR*(y_T/2+x_T*major/totalCN)+(1-TNR)*(y_T/2+x_T*minor/totalCN)
        p_N_minor = TPR*(y_N/2+x_N*minor/totalCN)+(1-TNR)*(y_N/2+x_N*major/totalCN)
        p_N_major = TPR*(y_N/2+x_N*major/totalCN)+(1-TNR)*(y_N/2+x_N*minor/totalCN)
        P_TA_E = np.logaddexp(
            log_binomial_pmf(C_TA, n=C_T, p=p_T_minor) - np.log(2), 
            log_binomial_pmf(C_TA, n=C_T, p=p_T_major) - np.log(2)
            )
        P_NA_E = np.logaddexp(
            log_binomial_pmf(C_NA, n=C_N, p=p_N_minor) - np.log(2), 
            log_binomial_pmf(C_NA, n=C_N, p=p_N_major) - np.log(2)
            )
        P_R_heterozygous = P_TA_E + P_NA_E
        # Clonal: 
        #P_TA_C, P_NA_C = 0, 0
        multiplicity = range(1, major + 1)        
        p_TA_C = [TPR*(x_T*i/totalCN) + (1-TNR)*(y_T+x_T*(totalCN-i)/totalCN) for i in multiplicity]
        p_NA_C = [TPR*(x_N*i/totalCN) + (1-TNR)*(y_N+x_N*(totalCN-i)/totalCN) for i in multiplicity]
        log_binoms_T = [log_binomial_pmf(C_TA, n=C_T, p=P) for P in p_TA_C]
        log_binoms_N = [log_binomial_pmf(C_NA, n=C_N, p=P) for P in p_NA_C]
        P_TA_C = logaddexp_reduce(log_binoms_T) - np.log(major)
        P_NA_C = logaddexp_reduce(log_binoms_N) - np.log(major)
        P_R_clonal = P_TA_C + P_NA_C

    # Combine the probabilities
    # set prior probabilities
    p_clonal = 0.0001
    p_germline = 1-p_clonal
    p_homozygous = AF*AF*p_germline
    p_heterozygous = 2*AF*(1-AF)*p_germline
    p_no_mutation = (1-AF)*(1-AF)*p_germline

    log_P_no_mutation = P_R_no_mutation + np.log(p_no_mutation)
    log_P_homozygous = P_R_homozygous + np.log(p_homozygous)
    log_P_heterozygous = P_R_heterozygous + np.log(p_heterozygous)
    log_P_clonal = P_R_clonal + np.log(p_clonal)

    log_P_R = logaddexp_reduce([log_P_no_mutation, log_P_homozygous, log_P_heterozygous, log_P_clonal])

    return -log_P_R


@njit
def objective(TiN_grid, data):
    best_TiN = 0
    min_neg_log_likelihood = np.inf
    for TiN in TiN_grid:
        negative_log_likelihood = 0
        for i in range(data_array.shape[0]):
            row = data_array[i]
            likelihood = calculate_probabilities(
                row[0], row[1], row[2], row[3],
                row[4], row[5], row[6], row[7], TiN
            )
            negative_log_likelihood += likelihood
        if negative_log_likelihood < min_neg_log_likelihood:
            min_neg_log_likelihood = negative_log_likelihood
            best_TiN = TiN
    return best_TiN, min_neg_log_likelihood

TiN_grid = np.linspace(0, 1, 1000)

data = pd.read_csv(
    #"/rsrch6/home/genetics/vanloolab/secure/TCGA/LUAD_WGS_BAM/TCGA-44-2655/partitioned/AFcombined/filtered_final.txt",
    f"/rsrch8/home/genetics/tchu/TCGA_LUAD/step6_estimate/subsamples/{sample}/random_sample{ID}.txt", 
    sep = "\t", header = 0, 
dtype={'chr': int, 'position': int, 'major': int, 'minor': int, 'totalCN': int, 'tumor_ref': int, 'tumor_alt': int, 'normal_ref': int, 'normal_alt': int, 'AF': float},
index_col=False)
data_array = data[['tumor_ref', 'tumor_alt', 'normal_ref', 'normal_alt', 'major', 'minor', 'totalCN', 'AF']].to_records(index=False)

optimal_TiN, min_neg_log_likelihood = objective(TiN_grid, data_array)

with open(f'{path}/estimated_outputs_{method}_{ID}.txt', 'w') as file:
    file.write(f"Optimal TiN: {optimal_TiN}\n")
    file.write(f"Minimum Negative Log Likelihood: {min_neg_log_likelihood}\n")
print("Optimal TiN is estimated as: ", optimal_TiN, "with corresponding neg llh: ", min_neg_log_likelihood)
