# Prior for TiN
alpha_tin <- 2
beta_tin <- 2
prior_TiN <- function(TiN) {
    dbeta(TiN, alpha_tin, beta_tin)
}

# Likelihood functions
TNR <- 0.72
TPR <- 0.95
likelihood_no_mutation <- function(C_NA, C_NR, C_TA, C_TR, TNR, TPR, TiT, TiN, VAF) {
    C <- C_NA + C_NR + C_TA + C_TR
    C_A <- C_NA + C_TA

    dbinom(C_A, C, 1 - TNR)
}
likelihood_homozygous <- function(C_NA, C_NR, C_TA, C_TR, TNR, TPR, TiT, TiN, VAF) {
    C <- C_NA + C_NR + C_TA + C_TR
    C_A <- C_NA + C_TA

    dbinom(C_A, C, TPR)
}
likelihood_heterozygous <- function(C_NA, C_NR, C_TA, C_TR, TNR, TPR, TiT, TiN, VAF) {
    C <- C_NA + C_NR + C_TA + C_TR
    C_A <- C_NA + C_TA
    p <- (1 - TNR + TPR) / 2

    dbinom(C_A, C, p)
}
likelihood_clonal <- function(C_NA, C_NR, C_TA, C_TR, TNR, TPR, TiT, TiN, VAF) {
    C_T <- C_TA + C_TR
    C_N <- C_NA + C_NR

    p_tumor <- TiT * ((1 - VAF) * (1 - TNR) + VAF * TPR) + (1 - TiT) * (1 - TNR)
    p_normal <- TiN * ((1 - VAF) * (1 - TNR) + VAF * TPR) + (1 - TiN) * (1 - TNR)

    tumor_likelihood <- dbinom(C_TA, C_T, p_tumor)
    normal_likelihood <- dbinom(C_NA, C_N, p_normal)
    return(tumor_likelihood * normal_likelihood)
}


# Overall
likelihood_overall <- function(C_NA, C_NR, C_TA, C_TR, TNR, TPR, TiT, TiN, VAF, theta) {
    no_mutation <- likelihood_no_mutation(C_NA, C_NR, C_TA, C_TR, TNR, TPR, TiT, TiN, VAF) * theta[1]
    heterozygous <- likelihood_heterozygous(C_NA, C_NR, C_TA, C_TR, TNR, TPR, TiT, TiN, VAF) * theta[2]
    homozygous <- likelihood_homozygous(C_NA, C_NR, C_TA, C_TR, TNR, TPR, TiT, TiN, VAF) * theta[3]
    clonal <- likelihood_clonal(C_NA, C_NR, C_TA, C_TR, TNR, TPR, TiT, TiN, VAF) * theta[4]
    
    return(no_mutation + heterozygous + homozygous + clonal)
}

posterior <- function(C_NA, C_NR, C_TA, C_TR, TNR, TPR, TiT, TiN, VAF, theta) {
    likelihood_value <- likelihood_overall(C_NA, C_NR, C_TA, C_TR, TNR, TPR, TiT, TiN, VAF, theta)
    prior_value <- prior_TiN(TiN)
    
    return(likelihood_value * prior_value)
}

optimize_TiN <- function(C_NA, C_NR, C_TA, C_TR, TNR, TPR, TiT, TiN, VAF, theta) {
    optim_res <- optimize(
        function(TiN) -posterior(C_NA, C_NR, C_TA, C_TR, TNR, TPR, TiT, TiN, VAF, theta),
        interval = c(0, 1), # TiN should be between 0 and 1
        maximum = FALSE
    )
    return(optim_res$minimum) # Estimated TiN value
}

