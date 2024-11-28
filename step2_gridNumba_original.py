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

THRESHOLD = 1e-300
@njit
def safe_log(prob):
    """Return log(prob) if prob > THRESHOLD, else -inf."""
    return np.log(prob) if prob > THRESHOLD else 0

@njit
def binomial_pmf(k, n, p):
    if k > n or k < 0:
        return 0.0
    coeff = 1.0
    for i in range(k):
        coeff *= (n - i) / (i + 1)
    return coeff * p**k * (1-p)**(n-k)

@njit
def calculate_probabilities(C_TR, C_TA, C_NR, C_NA, major, minor, totalCN, TiN, AF):
    C_T = C_TA + C_TR
    C_N = C_NA + C_NR
    C = C_T + C_N
    C_A = C_TA + C_NA
    if C == 0: 
        return 0.0

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
    # set prior probabilities
    if method == "uniform":
        p_clonal = 0.25
        p_homozygous = 0.25
        p_heterozygous = 0.25
        p_no_mutation = 0.25
    elif method == "byAF":
        p_clonal = 0.0001
        p_germline = 1-p_clonal
        p_homozygous = AF*AF*p_germline
        p_heterozygous = 2*AF*(1-AF)*p_germline
        p_no_mutation = (1-AF)*(1-AF)*p_germline

    log_P_no_mutation = safe_log(P_R_no_mutation * p_no_mutation)
    log_P_homozygous = safe_log(P_R_homozygous * p_homozygous)
    log_P_heterozygous = safe_log(P_R_heterozygous * p_heterozygous)
    log_P_clonal = safe_log(P_R_clonal * p_clonal)
    log_P_R = np.logaddexp(
        np.logaddexp(log_P_no_mutation, log_P_homozygous),
        np.logaddexp(log_P_heterozygous, log_P_clonal)
    )

    #print(P_no_mutation/P_R, P_homozygous/P_R, P_heterozygous/P_R, P_clonal/P_R)
    #print(C_TR, C_TA, C_NR, C_NA, major, minor)
    #print(P_R_no_mutation, P_R_homozygous, P_R_heterozygous, P_R_clonal)
    #print(P_R)
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
                row[4], row[5], row[6], TiN, row[7]
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
