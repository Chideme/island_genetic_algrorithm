import pandas as pd
import numpy as np
import riskfolio as rp


def HRP(returns):
    
    if np.isinf(returns).any().any():
        # Find where the infinite values are
        inf_mask = np.isinf(returns)
        inf_cols = returns.columns[inf_mask.any()]
        print("Columns with inf values:", inf_cols)

    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

    
    
    # 1. First check for zero variance columns
    zero_var_cols = returns.columns[returns.std() == 0]
    if len(zero_var_cols) > 0:
        print(f"Removing zero variance columns: {zero_var_cols}")
        returns = returns.drop(columns=zero_var_cols)
    port = rp.HCPortfolio(returns=returns)
    
    # 2. Calculate correlation matrix with handling for constant columns
    corr_matrix = returns.corr(method='pearson')
    
    # 3. Replace NaN correlations with 0
    corr_matrix = corr_matrix.fillna(0)
    
    # 4. Ensure the diagonal is 1
    np.fill_diagonal(corr_matrix.values, 1)
    
    model='HRP'
    codependence = 'pearson'
    rm = 'MV'
    rf = 0
    linkage = 'ward'
    max_k = 10
    leaf_order = True
    obj = 'Sharpe'
    
    # Convert correlation to distance matrix
    dist = np.sqrt(2*(1 - corr_matrix))
    
    # Verify the distance matrix
    print("Any NaN in final distance matrix:", np.isnan(dist).any().any())
    print("Any inf in final distance matrix:", np.isinf(dist).any().any())
    #Add small diagonal values to ensure positive definiteness
    epsilon = 1e-8
    cov_matrix = returns.cov()
    np.fill_diagonal(cov_matrix.values, cov_matrix.values.diagonal() + epsilon)
    
    model='HRP'
    codependence = 'pearson'
    rm = 'MV'
    rf = 0
    linkage = 'ward'
    max_k = 10
    leaf_order = True
    obj = 'Sharpe'
    
    w = port.optimization(model=model,
                        codependence=codependence,
                        rm=rm,
                        obj=obj,
                        rf=rf,
                        linkage=linkage,
                        max_k=max_k,
                        leaf_order=leaf_order,
                        custom_cov=cov_matrix)

    
    
    weights = w.T.values.tolist()[0]
    
    # If we removed any columns, add them back with zero weight
    if len(zero_var_cols) > 0:
        full_weights = pd.Series(0, index=returns.columns.union(zero_var_cols))
        full_weights[returns.columns] = weights
        weights = full_weights.tolist()
    
    return weights

def MV(returns):
    # MEAN VARIANCE MODEL
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    port = rp.Portfolio(returns=returns)

    # Choose the risk measure
    rm = 'MSV'  # Semi Standard Deviation

    # Estimate inputs of the model (historical estimates)
    method_mu='hist' # Method to estimate expected returns based on historical data.
    method_cov='hist' # Method to estimate covariance matrix based on historical data.

    port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)

    # Estimate the portfolio that maximizes the risk adjusted return ratio
    w = port.optimization(model='Classic', rm=rm, obj='Sharpe', rf=0.0, l=0, hist=True)
    weights = w.T.values.tolist()[0]
    return weights

def portfolio_return(returns,weights):
    cumulative_returns =  (1 + returns).cumprod().iloc[-1]-1
    portfolio_return = (cumulative_returns * weights).sum()
    return portfolio_return

def portfolio_std_dev(returns,weights):
    cov_matrix = returns.cov()
    portfolio_std_dev = np.sqrt(np.dot(weights, np.dot(cov_matrix.values, weights)))
    annualized_std_dev = portfolio_std_dev *  np.sqrt(252)
    return annualized_std_dev

def portfolio_sharpe_ratio(returns,weights):
   
    risk_free_rate = 0.00  # Replace with the actual risk-free rate
    # Calculate the portfolio returns
    portfolio_returns = np.dot(returns, weights)

    # Calculate the excess portfolio returns
    excess_portfolio_returns = portfolio_returns - risk_free_rate
    # Calculate the Sharpe Ratio
    sharpe_ratio = excess_portfolio_returns.mean() / excess_portfolio_returns.std()
    return sharpe_ratio

def calculate_mdd(weights,monthly_returns,portfolio):
    """Calculate the mdd for a Portfolio (TSP)"""
    weighted_returns = pd.DataFrame()
   
    for i, strategy in enumerate(portfolio):
        weighted_returns[strategy] = monthly_returns[strategy]* weights[i]
    # Calculate the combined weighted returns for the portfolio
    portfolio_returns = weighted_returns.sum(axis=1) 
    cumulative_returns = (1+ portfolio_returns ).cumprod()
    # Calculate the cumulative maximum value for each asset at the end of each month
    previous_peaks = cumulative_returns.cummax()
    drawdowns = (cumulative_returns - previous_peaks) / (1 + previous_peaks)
    mdd = drawdowns.min()
    return mdd





def portfolio_sortino_ratio(returns, weights):
    """
    Calculate the Sortino ratio for a portfolio.
    
    Parameters:
    -----------
    returns : numpy.ndarray
        Array of asset returns
    weights : numpy.ndarray
        Array of portfolio weights
    risk_free_rate : float, optional (default=0.00)
        The risk-free rate of return
    target_return : float, optional (default=0.0)
        Minimum acceptable return (MAR)
    periods_per_year : int, optional (default=252)
        Number of periods in a year (252 for daily data, 12 for monthly, etc.)
    
    Returns:
    --------
    float
        The annualized Sortino ratio
    """
    risk_free_rate = 0.00
    target_return = 0.0            
    periods_per_year = 252
    # Calculate portfolio returns
    portfolio_returns = np.dot(returns, weights)
    
    # Calculate excess returns above risk-free rate
    excess_returns = portfolio_returns - risk_free_rate
    
    # Calculate downside deviation
    downside_diff = np.where(portfolio_returns < target_return,
                            portfolio_returns - target_return,
                            0)
    
    # Calculate downside deviation (denominator of Sortino ratio)
    downside_std = np.sqrt(np.mean(downside_diff ** 2))
    

    # Handle case where there is no downside deviation
    if downside_std == 0:
        return np.inf if excess_returns > 0 else 0
    
    # Calculate annualized Sortino ratio
    sortino_ratio = np.sqrt(periods_per_year) * excess_returns.mean() / downside_std
    
    return sortino_ratio

def portfolio_calmar_ratio(returns,weights,mdd):
    """
    Calculate the Calmar ratio for a portfolio.
    
    Parameters:
    -----------
    returns : numpy.ndarray
        Array of asset returns
    weights : numpy.ndarray
        Array of portfolio weights
    risk_free_rate : float, optional (default=0.00)
        The risk-free rate of return
    periods_per_year : int, optional (default=252)
        Number of periods in a year (252 for daily data, 12 for monthly, etc.)
    
    Returns:
    --------
    float
        The annualized Calmar ratio
    """
    periods_per_year = 252
    # Calculate portfolio returns
    portfolio_returns = np.dot(returns, weights)
    
    # Calculate the maximum drawdown

    max_drawdown =  -mdd/100
    portfolio_mean = portfolio_returns.mean()
    
    # Calculate the annualized Calmar ratio
    calmar_ratio = np.sqrt(periods_per_year) * portfolio_mean / max_drawdown
    
    return calmar_ratio



def portfolio_metrics(assets,returns,weights):
   
  
    
    std_dev = round(portfolio_std_dev(returns,weights) * 100,4)
    sharpe_ratio = round(portfolio_sharpe_ratio(returns,weights),4)
    mdd =  round(calculate_mdd(weights,returns,assets)*100,4)
    portfolio_returns = round(portfolio_return(returns,weights) * 100,4)
    sortino_ratio = round(portfolio_sortino_ratio(returns, weights),4)
    calmar_ratio = round(portfolio_calmar_ratio(returns,weights,mdd),4)
    return sharpe_ratio, portfolio_returns, std_dev, mdd , sortino_ratio, calmar_ratio

  