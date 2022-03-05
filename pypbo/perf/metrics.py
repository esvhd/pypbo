import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy.stats as ss
import statsmodels.tsa.stattools as sts

# import statsmodels.tools.tools as stt
import statsmodels.distributions.empirical_distribution as sde

# import statsmodels.stats.diagnostic as ssd

import warnings

# maxzero = lambda x: np.maximum(x, 0)
# vmax = np.vectorize(np.maximum)

# default no. of trading days in a year, 252.
trading_days = 252


def _is_pandas(d):
    return isinstance(d, pd.DataFrame) or isinstance(d, pd.Series)


def _reindex_dates(source, target):
    """
    Reindex source data with target's index

    Parameters
    ----------
    source : TYPE
        data to reindex
    target : TYPE
        target data

    Returns
    -------
    TYPE
    """
    if _is_pandas(source) and _is_pandas(target):
        result = source.reindex(target.index)
    else:
        result = source
    # assert no NaN
    nan_flag = np.isnan(result)
    nan_check = nan_flag.sum()
    assert nan_check == 0, "Unmatched dates, NaN #{}".format(nan_check)
    return result


def log_excess(rtns, bench, debug=True):
    """
    Calculate excess return given two log return series.

    Args:
        rtns (TYPE): log returns
        bench (TYPE): benchmark log returns

    Returns:
        Log excess returns
    """
    # convert to pct space then back to log
    # if isinstance(rtns, pd.Series) or isinstance(rtns, pd.DataFrame):
    #     x = np.exp(rtns).sub(np.exp(bench), axis='index')
    # else:
    #     x = np.exp(rtns) - np.exp(bench)
    # y = 1 + x

    # if debug:
    #     valid_log = y > 0
    #     invalid_log = len(y) - np.sum(valid_log)
    #     debug_test = np.allclose(invalid_log, 0)
    #     # uncomment below to print count
    #     # print('log_excess debug: less than or close to 0 total = ',
    #     #       invalid_log)

    #     assert(debug_test), 'Log(0 or -ve) count = {}'.format(invalid_log)

    # excess = np.log(y)

    # first match return dates
    matched_bench = _reindex_dates(bench, rtns)
    excess = rtns - matched_bench
    return excess


def pct_to_log_excess(returns, bench):
    """
    Convert percentage returns to log returns, then compute log excess.

    Parameters
    ----------
    returns : TYPE

    bench : TYPE


    Returns
    -------
    TYPE
    """
    rtns_log = pct_to_log_return(returns)
    bench_log = pct_to_log_return(bench)
    return log_excess(rtns_log, bench_log)


def returns_gmean(returns):
    """
    Calculates geometric average returns from a given returns series.
    """
    if isinstance(returns, pd.DataFrame) or isinstance(returns, pd.Series):
        returns = returns.fillna(0)
    else:
        returns = np.nan_to_num(returns)
    return ss.gmean(1 + returns, axis=0) - 1


def log_returns(prices, n=1, fillna=False):
    """
    Log returns from prices. Preserves existing nan data when holidays are
    not aligned, i.e. return for date after an nan observation is done versus
    the last non-nan date.

    Parameters
    ----------
    prices : TYPE

    n : int, optional

    fillna : bool, optional
        If True fill first nan with 0.

    Returns
    -------
    TYPE
    """
    # keep null masks
    mask = prices.isnull()

    # ffill prices for calculating returns, one way to handle holiday calendars
    # the ffilled cells will be reset back to nan using the mask saved above.
    prices = prices.ffill()
    rtns = np.log(prices) - np.log(prices.shift(n))
    rtns.values[mask.values] = np.nan
    # print(rtns)

    if fillna:
        # rtns.fillna(0, inplace=True)
        # only fill first period nan
        if _is_pandas(rtns):
            rtns.values[0] = 0.0
        else:
            rtns[0] = 0.0
    return rtns


def pct_to_log_return(pct_returns, fillna=True):
    if _is_pandas(pct_returns):
        if fillna:
            pct_returns = pct_returns.fillna(0)
        return np.log(1 + pct_returns + 1e-8)
    else:
        if fillna:
            pct_returns = np.nan_to_num(pct_returns)
        return np.log(1 + pct_returns + 1e-8)


def log_to_pct_return(log_returns):
    return np.exp(log_returns) - 1


# def validate_mean_method(method):
#     if method not in {'gmean', 'simple'}:
#         raise AssertionError('mean_method can only be {"gmean", "simple"}')


# def validate_return_type(return_type):
#     if return_type not in {'pct', 'log'}:
#         raise AssertionError('mean_method can only be {"pct", "log"}')


def maxzero(x):
    return np.maximum(x, 0)


def LPM(returns, target_rtn, moment):
    """
    Lower partial moment.

    Parameters
    ----------
    returns : TYPE
        log returns
    target_rtn : TYPE

    moment : TYPE


    Returns
    -------
    TYPE
    """
    excess = -log_excess(returns, target_rtn)
    if _is_pandas(returns):
        # adj_returns = (target_rtn - returns).apply(maxzero)
        adj_returns = excess.clip(lower=0)
        return np.power(adj_returns, moment).mean()
    else:
        adj_returns = np.ndarray.clip(excess, min=0)
        return np.nanmean(np.power(adj_returns, moment), axis=0)


def kappa(returns, target_rtn, moment, log=True):
    """
    Geometric mean should be used when returns are percentage returns.
    Arithmetic mean should be used when returns are log returns.
    """
    # validate_return_type(return_type)

    if log:
        excess = log_excess(returns, target_rtn)
    else:
        # mean = returns_gmean(returns)
        # convert to log return then to log excess
        excess = pct_to_log_excess(returns, target_rtn)
        returns = pct_to_log_return(returns)
        target_rtn = pct_to_log_return(target_rtn)

    if _is_pandas(excess):
        mean = excess.mean()
    else:
        mean = np.nanmean(excess)

    kappa = mean / np.power(
        LPM(returns, target_rtn, moment=moment), 1.0 / moment
    )
    return kappa


def kappa3(returns, target_rtn=0, log=True):
    """
    Kappa 3
    """
    return kappa(returns, target_rtn=target_rtn, moment=3, log=log)


def omega(returns, target_rtn=0, log=True):
    """
    Omega Ratio
    """
    return 1 + kappa(returns, target_rtn=target_rtn, moment=1, log=log)


def omega_empirical(returns, target_rtn=0, log=True, plot=False, steps=1000):
    """
    Omega Ratio based on empirical distribution.
    """
    # validate_return_type(return_type)

    if not log:
        returns = pct_to_log_return(returns)

    # TODO
    ecdf = sde.ECDF(returns)

    # Generate computation space
    x = np.linspace(start=returns.min(), stop=returns.max(), num=steps)
    y = ecdf(x)

    norm_cdf = ss.norm.cdf(x, loc=returns.mean(), scale=returns.std(ddof=1))

    # Plot empirical distribution CDF versus Normal CDF with same mean and
    # stdev
    if plot:
        fig, ax = plt.subplots()
        fig.set_size_inches((12, 6))
        ax.plot(x, y, c="r", ls="--", lw=1.5, alpha=0.8, label="ECDF")
        ax.plot(x, norm_cdf, alpha=0.3, ls="-", c="b", lw=5, label="Normal CDF")
        ax.legend(loc="best")
        plt.show(fig)
        plt.close(fig)

    # TODO calculate omega ratio


def sortino(returns, target_rtn=0, factor=1, log=True):
    """
    Sortino I.I.D ratio caluclated using Lower Partial Moment.
    Result should be the same as `sortino_iid`.
    """
    # validate_return_type(return_type)

    if not log:
        excess = pct_to_log_excess(returns, target_rtn)
        returns = pct_to_log_return(returns)
    else:
        excess = log_excess(returns, target_rtn)

    if _is_pandas(returns):
        # return (returns.mean() - target_rtn) / \
        return (
            excess.mean()
            / np.sqrt(LPM(returns, target_rtn, 2))
            * np.sqrt(factor)
        )
    else:
        # return np.nanmean(returns - target_rtn) / \
        return (
            np.nanmean(excess)
            / np.sqrt(LPM(returns, target_rtn, 2))
            * np.sqrt(factor)
        )


def sortino_iid(rtns, bench=0, factor=1, log=True):
    # validate_return_type(return_type)

    if isinstance(rtns, np.ndarray):
        rtns = pd.DataFrame(rtns)

    if log:
        excess = log_excess(rtns, bench)
    else:
        excess = pct_to_log_excess(rtns, bench)

    neg_rtns = excess.where(cond=lambda x: x < 0)
    neg_rtns.fillna(0, inplace=True)
    semi_std = np.sqrt(neg_rtns.pow(2).mean())

    # print(excess, semi_std, np.std(neg_rtns, ddof=0))

    return np.sqrt(factor) * excess.mean() / semi_std


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


def match_rtn_dates(rtns, bench):
    if not (isinstance(rtns, pd.Series) or isinstance(rtns, pd.DataFrame)):
        # no need to reindex
        return bench

    if _is_pandas(bench):
        bench = bench.reindex(rtns.index)
        # check
        expected = len(rtns)
        check = bench.count()
        if expected != check:
            # warning
            warnings.warn(
                "Returns and benchmark length not matching, "
                "{} vs {}".format(expected, check)
            )
        return bench
    else:
        return bench


def sharpe_iid(rtns, bench=0, factor=1, log=True):
    """IID Sharpe ratio, percent returns are converted to log return.

    Parameters
    ----------
    rtns : TYPE

    bench : int, optional

    factor : int, optional

    log : bool, optional


    Returns
    -------
    TYPE
    """

    if log:
        excess = log_excess(rtns, bench)
    if not log:
        excess = pct_to_log_excess(rtns, bench)

    # print('excess: ', excess)

    if _is_pandas(rtns):
        excess_mean = excess.mean()
        return np.sqrt(factor) * excess_mean / excess.std(ddof=1)
    else:
        # numpy way
        excess_mean = np.nanmean(excess, axis=0)
        return np.sqrt(factor) * excess_mean / np.nanstd(excess, axis=0, ddof=1)


def sharpe_iid_rolling(
    rtns, window: int, min_periods: int, bench=0, factor=1, log=True
):
    """
    Rolling sharpe ratio, unadjusted by time aggregation.
    """
    # validate_return_type(return_type)

    if log:
        excess = log_excess(rtns, bench)
    else:
        excess = pct_to_log_excess(rtns, bench)

    roll = excess.rolling(window=window, min_periods=min_periods)
    return np.sqrt(factor) * roll.mean() / roll.std(ddof=1)


def sharpe_iid_adjusted(rtns, bench=0, factor=1, log=True):
    """
    Adjusted Sharpe Ratio, acount for skew and kurtosis in return series.

    Pezier and White (2006) adjusted sharpe ratio.

    https://www.google.co.uk/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=0ahUKEwi42ZKgg_TOAhVFbhQKHSXPDY0QFggcMAA&url=http%3A%2F%2Fwww.icmacentre.ac.uk%2Fpdf%2Fdiscussion%2FDP2006-10.pdf&usg=AFQjCNF9axYf4Gbz4TVdJUdM8o2M2rz-jg&sig2=pXHZ7M-n-PtNd2d29xhRBw

    Parameters:
        rtns:
            returns dataframe. Default should be log returns
        bench:
            benchmark return
        factor:
            time aggregation factor, default 1, i.e. not adjusted.
        log (bool, optional):
            log return or not, default True

    Deleted Parameters:
        return_type: {'log', 'pct'}, return series type, log or arithmetic
            percentages.

    Returns:
        TYPE
    """
    sr = sharpe_iid(rtns, bench=bench, factor=1, log=log)
    # print(sr)

    if _is_pandas(rtns):
        skew = rtns.skew()
        excess_kurt = rtns.kurtosis()
    else:
        skew = ss.skew(rtns, bias=False, nan_policy="omit")
        excess_kurt = ss.kurtosis(
            rtns, bias=False, fisher=True, nan_policy="omit"
        )
    return adjusted_sharpe(sr, skew, excess_kurt) * np.sqrt(factor)


def adjusted_sharpe(sr, skew, excess_kurtosis):
    """
    Adjusted Sharpe Ratio, acount for skew and kurtosis in return series.

    Pezier and White (2006) adjusted sharpe ratio.

    https://www.google.co.uk/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=0ahUKEwi42ZKgg_TOAhVFbhQKHSXPDY0QFggcMAA&url=http%3A%2F%2Fwww.icmacentre.ac.uk%2Fpdf%2Fdiscussion%2FDP2006-10.pdf&usg=AFQjCNF9axYf4Gbz4TVdJUdM8o2M2rz-jg&sig2=pXHZ7M-n-PtNd2d29xhRBw

    Parameters:
        sr :
            sharpe ratio
        skew :
            return series skew
        excess_kurtosis :
            return series excess kurtosis
    """
    # return sr * (1 + (skew / 6.0) * sr + (kurtosis - 3) / 24.0 * sr**2)
    return sr * (1 + (skew / 6.0) * sr + excess_kurtosis / 24.0 * sr ** 2)


def sharpe_non_iid(rtns, bench=0, q=trading_days, p_critical=0.05, log=True):
    """
    Return Sharpe Ratio adjusted for auto-correlation, iff Ljung-Box test
    indicates that the return series exhibits auto-correlation. Based on
    Andrew Lo (2002).

    Parameters:
        rtns:
            return series
        bench:
            risk free rate, default 0
        q:
            time aggregation frequency, e.g. 12 for monthly to annual.
            Default 252.
        p_critical:
            critical p-value to reject Ljung-Box Null, default 0.05.
        log (bool, optional):
            True if rtns is log returns, default True

    Deleted Parameters:
        return_type:
            {'log', 'pct'}, return series type, log or arithmetic
            percentages.

    Returns:
        TYPE
    """
    if type(q) is not np.int64 or type(q) is not np.int32:
        q = np.round(q, 0).astype(np.int64)

    if len(rtns) <= q:
        # raise AssertionError('No. of returns [{}] must be greated than {}'
        #                      .format(len(rtns), q))
        warnings.warn(
            "Sharpe Non-IID: No. of returns [{}] must be greater"
            " than {}. NaN returned.".format(len(rtns), q)
        )
        dim = rtns.shape
        if len(dim) < 2:
            return np.nan
        else:
            res = np.empty((1, dim[1]))
            res[:] = np.nan
            return res

    sr = sharpe_iid(rtns, bench=bench, factor=1, log=log)

    if not _is_pandas(rtns):
        adj_factor, pval = sharpe_autocorr_factor(rtns, q=q)
        if pval < p_critical:
            # reject Ljung-Box Null, there is serial correlation
            return sr * adj_factor
        else:
            return sr * np.sqrt(q)
    else:
        if isinstance(rtns, pd.Series):
            tests = [sharpe_autocorr_factor(rtns.dropna().values, q=q)]
        else:
            tests = [
                sharpe_autocorr_factor(rtns[col].dropna().values, q=q)
                for col in rtns.columns
            ]
        factors = [
            adj_factor if pval < p_critical else np.sqrt(q)
            for adj_factor, pval in tests
        ]

        if isinstance(rtns, pd.Series):
            out = sr * factors[0]
        else:
            res = pd.Series(factors, index=rtns.columns)
            out = res.multiply(sr)

        return out


def sharpe_autocorr_factor(returns, q):
    """
    Auto-correlation correction for Sharpe ratio time aggregation based on
    Andrew Lo's 2002 paper.

    Link:
    https://www.google.co.uk/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=0ahUKEwj5wf2OjO_OAhWDNxQKHT0wB3EQFggeMAA&url=http%3A%2F%2Fedge-fund.com%2FLo02.pdf&usg=AFQjCNHbSz0LDZxFXm6pmBQukCfAYd0K7w&sig2=zQgZAN22RQcQatyP68VKmQ

    Parameters:
        returns :
            return sereis
        q :
            time aggregation factor, e.g. 12 for monthly to annual,
            252 for daily to annual

    Returns:
        factor : time aggregation factor
        p-value : p-value for Ljung-Box serial correation test.
    """
    # Ljung-Box Null: data is independent, i.e. no auto-correlation.
    # smaller p-value would reject the Null, i.e. there is auto-correlation
    acf, _, pval = sts.acf(returns, adjusted=False, nlags=q, qstat=True)
    term = [(q - (k + 1)) * acf[k + 1] for k in range(q - 2)]
    factor = q / np.sqrt(q + 2 * np.sum(term))

    return factor, pval[-2]


def annual_geometric_returns(rtns, ann_factor=trading_days, log=True):
    """
    Take a return series and produce annualized geometric return.

    Args:
        rtns (TYPE):
            return series, log or pct returns
        ann_factor (TYPE, optional):
            annual day count factor
        log (bool, optional):
            True if log return is given. Default True.

    Returns:
        float, annualized geometric return
    """
    if not log:
        rtns = pct_to_log_return(rtns)
    total_rtn = np.exp(rtns.sum())
    geo = np.power(total_rtn, ann_factor / len(rtns)) - 1
    return geo


def annualized_pct_return(total_return, days, ann_factor=trading_days):
    """
    Parameters:
        total_return:
            total pct equity curve, e.g. if return is +50%, then this
            should be 1.5 (e.g. 1. + .5)
        days :
            number of days in period.
        ann_factor :
            number of days in a year
    Returns:
        Annualized percentage return.
    """
    years = days / ann_factor
    ann = np.power(total_return, 1.0 / years) - 1
    return ann


def annualized_log_return(total_return, days, ann_factor=trading_days):
    """
    Parameters:
        total_return:
            total log return, e.g. if return is +50%, then this should be 0.5,
            e.g. not 1.5.
        days :
            number of days in period.
        ann_factor :
            number of days in a year
    Returns:
        Annualized percentage return.
    """

    years = days / ann_factor
    ann = total_return / years
    return ann


def tail_ratio(returns, tail_prob=5):
    """
    Determines the ratio between the right (95%) and left tail (5%).
    For example, a ratio of 0.25 means that losses are four times
    as bad as profits.

    Parameters
    ----------
    returns : pd.Series
        asset returns
    tail_prob : int, optional
        in the range of [0, 100], to match numpy.nanpercentile()

    Returns
    -------
    float
        tail_prob ratio
    """
    if _is_pandas(returns):
        tail_prob /= 100.0
        top = returns.quantile(q=1 - tail_prob)
        bottom = returns.quantile(q=tail_prob)
        return np.abs(top / bottom)
    else:
        return np.abs(np.nanpercentile(returns, 100 - tail_prob)) / np.abs(
            np.nanpercentile(returns, tail_prob)
        )


def max_drawdown(equity):
    return drawdown(equity).min()


def max_drawdown_from_rtns(returns, log=True):
    return drawdown_from_rtns(returns, log=log).min()


def drawdown(equity) -> pd.DataFrame:
    """
    Drawdown curve.

    Args:
        equity (DataFrame or Series/Array like):
            equity curve

    Returns:
        drawdown curve in percentage terms from peaks.
    """
    if isinstance(equity, np.ndarray) or isinstance(equity, list):
        equity = pd.DataFrame(equity)
    highs = equity.expanding().max()
    dd = equity / highs - 1.0
    return dd


def drawdown_from_rtns(returns, log=True):
    """
    Drowdown curve from returns.

    Args:
        returns (array like):
            asset returns
        log (bool, optional):
            log returns or not. Default True

    Returns:
        TYPE
    """
    if log:
        equity = np.exp(returns.cumsum())
    else:
        equity = (1 + returns).cumprod()
    return drawdown(equity)


def calmar_ratio(returns, factor=trading_days, log=True):
    """
    CALMAR ratio: annualized return over  max drawdown, for last 36
    months.

    See Wikipedia: https://en.wikipedia.org/wiki/Calmar_ratio

    Parameters:
        returns :
            return series
    Returns:
        Calmar ratio, calculated with normal percentage returns.
    """
    if not log:
        returns = pct_to_log_return(returns)

    num_years = float(len(returns)) / factor

    if not log:
        cum_return = (1 + returns).cumprod()
    else:
        # log return
        cum_return = np.exp(returns.cumsum())

    if isinstance(cum_return, np.ndarray) or isinstance(cum_return, list):
        cum_return = pd.Series(cum_return)

    annual_return = np.power(cum_return.values[-1], 1 / num_years) - 1

    # max_dd = np.abs(get_drawdown(cum_return)['drawdown'].min())
    max_dd = np.abs(drawdown(cum_return).min())

    return annual_return / max_dd
