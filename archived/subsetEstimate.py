import numpy as np
import pandas as pd
from scipy.stats import binom
from scipy.optimize import minimize

data = pd.read_csv("/rsrch8/home/genetics/tchu/TCGA_LUAD/estimate/random_sample.txt", sep = "\t", header = 0, 
dtype={'chr': int, 'position': int, 'major': int, 'minor': int, 'totalCN': int, 'tumor_ref': int, 'tumor_alt': int, 'normal_ref': int, 'normal_alt': int},
index_col=False)

# Set up parameters
TPR = 0.9
  # alternate and seq'd as alternate
TNR = 0.95  # reference and seq'd as reference
TiT = 0.27

# Prior probabilities of four conditions
p_homozygous = 0.0001 # prior probability of homozygous germline mutation
p_heterozygous = 0.001 # prior probability of heterozygous germline mutation
p_clonal = 0.0001      # prior probability of clonal mutation
p_no_mutation = 1 - p_homozygous - p_heterozygous - p_clonal # add up to 1

def calculate_probabilities(C_TR, C_TA, C_NR, C_NA, major, minor, totalCN, TiN):
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
    #print(x_T, x_N)
    
    # Calculate likelihoods
    # No mutation:
    #P_TR_N = binom.pmf(C_TR, n=C_T, p=TNR)
    P_TA_N = binom.pmf(C_TA, n=C_T, p=1-TNR)
    #P_NR_N = binom.pmf(C_NR, n=C_N, p=TNR)
    P_NA_N = binom.pmf(C_NA, n=C_N, p=1-TNR)
    #P_R_no_mutation = P_TR_N * P_TA_N * P_NR_N * P_NA_N
    P_R_no_mutation = P_TA_N * P_NA_N

    # Homozygous germline:
    #P_TR_O = binom.pmf(C_TR, n=C_T, p=1-TPR)
    P_TA_O = binom.pmf(C_TA, n=C_T, p=TPR)
    #P_NR_O = binom.pmf(C_NR, n=C_N, p=1-TPR)
    P_NA_O = binom.pmf(C_NA, n=C_N, p=TPR)
    #P_R_homozygous = P_TR_O * P_TA_O * P_NR_O * P_NA_O
    P_R_homozygous = P_TA_O * P_NA_O

    if totalCN == 0:
        #P_TR_E = binom.pmf(C_TR, n=C_T, p=y*(1-TPR+TNR)/2)
        P_TA_E = binom.pmf(C_TA, n=C_T, p=y_T*(TPR+1-TNR)/2)
        #P_NR_E = binom.pmf(C_NR, n=C_N, p=y*(1-TPR+TNR)/2)
        P_NA_E = binom.pmf(C_NA, n=C_N, p=y_N*(TPR+1-TNR)/2)
        #P_R_heterozygous = P_TR_E * P_TA_E * P_NR_E * P_NA_E
        P_R_heterozygous = P_TA_E * P_NA_E

        #P_TR_C = binom.pmf(C_TR, n=C_T, p=y*TNR)
        P_TA_C = binom.pmf(C_TA, n=C_T, p=y_T*(1-TNR))
        #P_NR_C = binom.pmf(C_NR, n=C_N, p=TNR)
        P_NA_C = binom.pmf(C_NA, n=C_N, p=y_N*(1-TNR))
        #P_R_clonal = P_TR_C * P_TA_C * P_NR_C * P_NA_C
        P_R_clonal = P_TA_C * P_NA_C

    else:
        # Heterozygous germline:
        #p1=((1-TPR)*(y/2+x*minor/totalCN)+TNR*(y/2+x*major/totalCN) + (1-TPR)*(y/2+x*major/totalCN)+TNR*(y/2+x*minor/totalCN))/2
        p_T_minor = TPR*(y_T/2+x_T*minor/totalCN)+(1-TNR)*(y_T/2+x_T*major/totalCN)
        p_T_major = TPR*(y_T/2+x_T*major/totalCN)+(1-TNR)*(y_T/2+x_T*minor/totalCN)
        p_N_minor = TPR*(y_N/2+x_N*minor/totalCN)+(1-TNR)*(y_N/2+x_N*major/totalCN)
        p_N_major = TPR*(y_N/2+x_N*major/totalCN)+(1-TNR)*(y_N/2+x_N*minor/totalCN)
        #P_TR_E = binom.pmf(C_TR, n=C_T, p=p1)
        P_TA_E = binom.pmf(C_TA, n=C_T, p=p_T_minor)/2 + binom.pmf(C_TA, n=C_T, p=p_T_major)/2
        #P_NR_E = binom.pmf(C_NR, n=C_N, p=p1)
        P_NA_E = binom.pmf(C_NA, n=C_N, p=p_N_minor)/2 + binom.pmf(C_NA, n=C_N, p=p_N_major)/2
        #P_R_heterozygous = P_TR_E * P_TA_E * P_NR_E * P_NA_E
        #print(p_T_minor, p_N_minor)
        P_R_heterozygous = P_TA_E * P_NA_E

        # Clonal: 
        P_TA_C, P_NA_C = 0, 0
        multiplicity = np.arange(1, major + 1)
        for i in multiplicity:
            p_TA_C = x_T*(TPR*i/totalCN+(1-TNR)*(totalCN-i)/totalCN)+y_T*(1-TNR)
            p_NA_C = x_N*(TPR*i/totalCN+(1-TNR)*(totalCN-i)/totalCN)+y_N*(1-TNR)
            #P_TR_C = binom.pmf(C_TR, n=C_T, p=p_TR_C)
            P_TA_C += binom.pmf(C_TA, n=C_T, p=p_TA_C)/major
            #P_NR_C = binom.pmf(C_NR, n=C_N, p=TiN*(1-TPR)+(1-TiN)*TNR)
            P_NA_C += binom.pmf(C_NA, n=C_N, p=p_NA_C)/major
        #P_R_clonal = P_TR_C * P_TA_C * P_NR_C * P_NA_C
        P_R_clonal = P_TA_C * P_NA_C

    # Combine the probabilities
    P_R = (P_R_no_mutation * p_no_mutation + 
            P_R_homozygous * p_homozygous + 
            P_R_heterozygous * p_heterozygous + 
            P_R_clonal * p_clonal)
    #print(C_TR, C_TA, C_NR, C_NA, major, minor)
    #print(P_R_no_mutation, P_R_homozygous, P_R_heterozygous, P_R_clonal)
    #print(P_R)
    return -np.log(P_R)


def objective(TiN):
    negative_log_likelihood = 0
    for _, row in data.iterrows():
        likelihood = calculate_probabilities(
            row['tumor_ref'], row['tumor_alt'], row['normal_ref'], row['normal_alt'],
            row['major'], row['minor'], row['totalCN'], TiN
        )
        negative_log_likelihood += likelihood
    return negative_log_likelihood

# Run the optimization to find the best TiN
result = minimize(objective, 0, bounds=[(0, 1)], method='L-BFGS-B')

# Output the optimized TiN value
optimal_TiN = result.x[0]
print(f"Optimal TiN: {optimal_TiN}")