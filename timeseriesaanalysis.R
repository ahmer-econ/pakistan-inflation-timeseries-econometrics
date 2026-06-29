# ==============================================================================
# Script: Comprehensive Time-Series Macroeconometric Analysis
# Description: Unit Root tests, ARIMA, VAR, Granger Causality, IRFs, and ARCH-LM.
# ==============================================================================

# Install necessary packages if you haven't already:
# install.packages(c("dplyr", "readr", "tseries", "forecast", "vars", "FinTS"))

library(dplyr)
library(readr)
library(tseries)   # For Augmented Dickey-Fuller (ADF) tests
library(forecast)  # For Box-Jenkins ARIMA modeling
library(vars)      # For VAR, Granger Causality, and IRFs
library(FinTS)     # For ARCH-LM volatility clustering tests

# ------------------------------------------------------------------------------
# 1. Data Import & Transformation
# ------------------------------------------------------------------------------
# Load the raw dataset
# (Update the file path to match your repository structure)
df <- read_csv("data/pakistan_macro_data.csv")

# Sort by year and calculate log returns/growth rates for stationarity
df_ts <- df %>%
  arrange(Year) %>%
  mutate(
    log_m2   = log(money_supply_m2),
    log_govt = log(govt_expenditure),
    log_ex   = log(exchange_rate),
    
    m2_growth   = log_m2 - lag(log_m2),
    govt_growth = log_govt - lag(log_govt),
    ex_growth   = log_ex - lag(log_ex)
  ) %>%
  na.omit()

# Convert the cleaned columns into formal time-series (ts) objects
# (Assuming annual data starting from 1970 based on EViews output)
start_year <- min(df_ts$Year)
cpi_ts  <- ts(df_ts$cpi_inflation, start = start_year, frequency = 1)
m2_ts   <- ts(df_ts$m2_growth, start = start_year, frequency = 1)
govt_ts <- ts(df_ts$govt_growth, start = start_year, frequency = 1)
ex_ts   <- ts(df_ts$ex_growth, start = start_year, frequency = 1)

# ------------------------------------------------------------------------------
# 2. Stationarity Testing (Augmented Dickey-Fuller)
# ------------------------------------------------------------------------------
print("=== ADF Test for Stationarity ===")
# Testing the null hypothesis of a unit root (non-stationarity)
adf.test(cpi_ts)
adf.test(ex_ts)

# ------------------------------------------------------------------------------
# 3. ARIMA Modeling (Box-Jenkins on CPI Inflation)
# ------------------------------------------------------------------------------
print("=== ARIMA Model: AR(2) on CPI Inflation ===")
# We identified an AR(2) model as the best fit based on AIC parsimony
cpi_arima <- Arima(cpi_ts, order = c(2, 0, 0))
summary(cpi_arima)

# Check residuals to ensure they are strictly white noise (Ljung-Box test)
checkresiduals(cpi_arima)

# ------------------------------------------------------------------------------
# 4. VAR Modeling (Vector Autoregression)
# ------------------------------------------------------------------------------
# Bind the stationary variables into a single multivariate system
var_data <- cbind(cpi_ts, m2_ts, govt_ts, ex_ts)
colnames(var_data) <- c("CPI_INFLATION", "M2_GROWTH", "GOVT_GROWTH", "EX_GROWTH")

print("=== VAR Lag Length Selection ===")
# Run information criteria tests to determine optimal memory
lag_selection <- VARselect(var_data, lag.max = 4, type = "const")
print(lag_selection$selection)

print("=== VAR(1) Model Estimation ===")
# Estimate the VAR(1) model based on the AIC winner
var_model <- VAR(var_data, p = 1, type = "const")
summary(var_model)

# ------------------------------------------------------------------------------
# 5. Granger Causality (Block Exogeneity) Tests
# ------------------------------------------------------------------------------
print("=== Granger Causality Tests ===")
# Testing if M2 Growth causes Exchange Rate Depreciation
causality(var_model, cause = "M2_GROWTH")$Granger

# Testing if Government Growth causes Exchange Rate shifts
causality(var_model, cause = "GOVT_GROWTH")$Granger

# ------------------------------------------------------------------------------
# 6. Impulse Response Functions (IRFs)
# ------------------------------------------------------------------------------
# Generate orthogonalized impulse responses (10 periods ahead)
irf_model <- irf(var_model, n.ahead = 10, boot = TRUE, ci = 0.95)

# Plot the full 16-chart matrix of IRFs
# (This will render the visual grid showing shock transmissions over time)
plot(irf_model)

# ------------------------------------------------------------------------------
# 7. ARCH-LM Test (Testing for Volatility Clustering)
# ------------------------------------------------------------------------------
print("=== ARCH-LM Test on Exchange Rate Growth ===")
# Build a simple mean equation (constant only) for the exchange rate
ex_mean_model <- lm(ex_ts ~ 1) 
ex_residuals <- residuals(ex_mean_model)

# Run the Lagrange Multiplier test for ARCH effects with 4 lags
arch_test <- ArchTest(ex_residuals, lags = 4, demean = FALSE)
print(arch_test)

# Result Note: Annual data typically yields p > 0.05 (No ARCH effects).
# GARCH models are reserved for daily/high-frequency financial data.