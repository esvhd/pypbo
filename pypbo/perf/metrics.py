import numpy as np
import pandas as pd

import scipy.stats as ss
import statsmodels.tsa.stattools as sts
# import statsmodels.stats.diagnostic as ssd


# maxzero = lambda x: np.maximum(x, 0)
# vmax = np.vectorize(np.maximum)

trading_days = 255


def returns_gmean(returns):
    '''
    Calculates geometric average returns from a given returns series.
    '''
    if isinstance(returns, pd.DataFrame) or isinstance(returns, pd.Series):
        returns = returns.fillna(0)
    else:
        returns = np.nan_to_num(returns)
    return ss.gmean(1 + returns, axis=0) - 1


def log_returns(prices):
    return np.log(prices / prices.shift(1)).fillna(0)


def pct_to_log_return(pct_returns):
    return np.log(1 + pct_returns.fillna(0))


def log_to_pct_return(log_returns):
    return np.exp(log_returns) - 1


def validate_mean_method(method):
    if method not in {'gmean', 'simple'}:
        raise AssertionError('mean_method can only be {"gmean", "simple"}')


def validate_return_type(return_type):
    if return_type not in {'pct', 'log'}:
        raise AssertionError('mean_method can only be {"pct", "log"}')


def maxzero(x):
    return np.maximum(x, 0)


def LPM(returns, target_rtn, moment):
    if isinstance(returns, pd.DataFrame) or isinstance(returns, pd.Series):
        adj_returns = (target_rtn - returns).apply(maxzero)
        return np.power(adj_returns, moment).mean()
    else:
        adj_returns = np.ndarray.clip(target_rtn - returns, min=0)
        # only averaging over non nan values
        # return np.nansum(np.power(adj_returns, moment)) / \
        #     np.count_nonzero(~np.isnan(adj_returns))
        return np.nanmean(np.power(adj_returns, moment), axis=0)


def kappa(returns, target_rtn, moment, return_type='log'):
    '''
    Geometric mean should be used when returns are percentage returns.
    Arithmetic mean should be used when returns are log returns.
    '''
    validate_return_type(return_type)

    if return_type == 'pct':
        mean = returns_gmean(returns)
    else:
        if isinstance(returns, pd.DataFrame) or isinstance(returns, pd.Series):
            mean = returns.mean()
        else:
            mean = np.nanmean(returns)

    kappa = (mean - target_rtn) / np.power(LPM(returns,
                                               target_rtn,
                                               moment=moment),
                                           1.0 / moment)
    return kappa


def kappa3(returns, target_rtn=0, return_type='log'):
    '''
    Kappa 3
    '''
    return kappa(returns, target_rtn=target_rtn, moment=3,
                 return_type=return_type)


def omega(returns, target_rtn=0, return_type='log'):
    '''
    Omega Ratio
    '''
    return 1 + kappa(returns,
                     target_rtn=target_rtn,
                     moment=1,
                     return_type=return_type)


def sortino(returns, target_rtn=0, factor=1, return_type='log'):
    validate_return_type(return_type)

    if return_type == 'log':
        if isinstance(returns, pd.DataFrame) or isinstance(returns, pd.Series):
            return (returns.mean() - target_rtn) / \
                np.sqrt(LPM(returns, target_rtn, 2))
        else:
            return np.nanmean(returns - target_rtn) / \
                np.sqrt(LPM(returns, target_rtn, 2))
    else:
        mean = returns_gmean(returns)
        return (mean - target_rtn) / \
            np.sqrt(LPM(returns, target_rtn, 2)) * factor


def sortino_iid(df, bench=0, factor=1, return_type='log'):
    validate_return_type(return_type)

    if isinstance(df, np.ndarray):
        df = pd.DataFrame(df)

    if return_tpye == 'pct':
        returns = returns_gmean(df) - bench
    else:
        returns = df.mean() - bench

    # neg_rtns = df.loc[df < 0]
    neg_rtns = df.where(cond=lambda x: x < 0)
    semi_std = neg_rtns.std(ddof=1)
    return factor * returns / semi_std


# def rolling_lpm(returns, target_rtn, moment, window):
#     adj_returns = returns - target_rtn
#     adj_returns[adj_returns > 0] = 0
#     return pd.rolling_mean(adj_returns**moment,
#                            window=window, min_periods=window)


# def rolling_sortino(returns, window, target_rtn=0):
#     '''
#     This is ~150x faster than using rolling_ratio which uses rolling_apply
#     '''
#     num = pd.rolling_mean(returns, window=window,
#                           min_periods=window) - target_rtn
#     denom = np.sqrt(rolling_lpm(returns, target_rtn,
#                                 moment=2, window=window))
#     return num / denom


# def sharpe(returns, bench_rtn=0):
#     excess = returns - bench_rtn
#     if isinstance(excess, pd.DataFrame) or isinstance(excess, pd.Series):
#         return excess.mean() / excess.std(ddof=1)
#     else:
#         return np.nanmean(excess) / np.nanstd(excess, ddof=1)


def sharpe_iid(df, bench=0, factor=1, return_type='log'):
    validate_return_type(return_type)

    excess = df - bench
    if isinstance(df, pd.DataFrame) or isinstance(df, pd.Series):
        # return factor * excess.mean() / excess.std(ddof=1)
        if return_type == 'pct':
            excess_mean = returns_gmean(excess)
        else:
            excess_mean = excess.mean()
        return factor * excess_mean / excess.std(ddof=1)
    else:
        # numpy way
        if return_type == 'pct':
            excess_mean = returns_gmean(excess)
        else:
            excess_mean = np.nanmean(excess, axis=0)
        return factor * excess_mean / np.nanstd(excess, axis=0, ddof=1)


def sharpe_iid_rolling(df, window, bench=0, factor=1):
    '''
    Not using geometric mean returns.
    '''
    roll = (df - bench).rolling(window=window)
    # return factor * (roll.mean() - bench) / roll.std(ddof=1)
    return factor * roll.mean() / roll.std(ddof=1)


def sharpe_iid_adjusted(df, bench=0, factor=1, return_type='log'):
    '''
    Adjusted Sharpe Ratio, acount for skew and kurtosis in return series.

    Pezier and White (2006) adjusted sharpe ratio.

    https://www.google.co.uk/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=0ahUKEwi42ZKgg_TOAhVFbhQKHSXPDY0QFggcMAA&url=http%3A%2F%2Fwww.icmacentre.ac.uk%2Fpdf%2Fdiscussion%2FDP2006-10.pdf&usg=AFQjCNF9axYf4Gbz4TVdJUdM8o2M2rz-jg&sig2=pXHZ7M-n-PtNd2d29xhRBw
    '''
    sr = sharpe_iid(df, bench, factor=1, return_type=return_type)

    if isinstance(df, pd.DataFrame) or isinstance(df, pd.Series):
        skew = df.skew()
        excess_kurt = df.kurtosis()
    else:
        skew = ss.skew(df, bias=False, nan_policy='omit')
        excess_kurt = ss.kurtosis(df, bias=False, fisher=True,
                                  nan_policy='omit')
    return adjusted_sharpe(sr, skew, excess_kurt) * factor


def adjusted_sharpe(sr, skew, excess_kurtosis):
    # return sr * (1 + (skew / 6.0) * sr + (kurtosis - 3) / 24.0 * sr**2)
    return sr * (1 + (skew / 6.0) * sr + excess_kurtosis / 24.0 * sr**2)


def sharpe_non_iid(df, bench=0, q=trading_days, p_critical=.05):
    '''
    Return Sharpe Ratio adjusted for auto-correlation, iff Ljung-Box test
    indicates that the return series exhibits auto-correlation. Based on
    Andrew Lo (2002).

    Parameters:
        df : return series
        bench : risk free rate, default 0
        q : time aggregation frequency, e.g. 12 for monthly to annual.
            Default 255.
        p_critical : critical p-value to reject Ljung-Box Null, default 0.05.
    '''
    sr = sharpe_iid(df, bench=bench, factor=1)

    if not isinstance(df, pd.DataFrame):
        adj_factor, pval = sharpe_autocorr_factor(df, q=q)
        if pval < p_critical:
            # reject Ljung-Box Null, there is serial correlation
            return sr * adj_factor
        else:
            return sr * np.sqrt(q)
    else:
        tests = [sharpe_autocorr_factor(df[col].dropna().values, q=q)
                 for col in df.columns]
        factors = [adj_factor if pval < p_critical else np.sqrt(q)
                   for adj_factor, pval in tests]
        res = pd.Series(factors, index=df.columns)

        return res.multiply(sr)


def sharpe_autocorr_factor(returns, q):
    '''
    Auto-correlation correction for Sharpe ratio time aggregation based on
    Andrew Lo's 2002 paper.

    Link:
    https://www.google.co.uk/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=0ahUKEwj5wf2OjO_OAhWDNxQKHT0wB3EQFggeMAA&url=http%3A%2F%2Fedge-fund.com%2FLo02.pdf&usg=AFQjCNHbSz0LDZxFXm6pmBQukCfAYd0K7w&sig2=zQgZAN22RQcQatyP68VKmQ

    Parameters:
        returns : return sereis
        q : time aggregation factor, e.g. 12 for monthly to annual,
        255 for daily to annual

    Returns:
        factor : time aggregation factor
        p-value : p-value for Ljung-Box serial correation test.
    '''
    # Ljung-Box Null: data is independent, i.e. no auto-correlation.
    # smaller p-value would reject the Null, i.e. there is auto-correlation
    acf, _, pval = sts.acf(returns, unbiased=False, nlags=q, qstat=True)
    term = [(q - (k + 1)) * acf[k + 1] for k in range(q - 2)]
    factor = q / np.sqrt(q + 2 * np.sum(term))

    return factor, pval[-2]


if __name__ == 'main':
    pass
