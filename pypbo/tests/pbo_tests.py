import pytest
import numpy as np

import pypbo as pbo


def test_expected_max():
    assert np.isclose(pbo.expected_max(100), 2.5306028932016846)


def test_psr():
    '''
    Example from The Sharpe Ratio Efficient Frontier
    '''
    assert_val = 0.91323430688486018

    sr = 1.585 / np.sqrt(12)
    skew = -2.448
    kurt = 10.164
    T = 24

    stat = pbo.psr(sr, T=T, skew=skew, kurtosis=kurt, target_sharpe=0)

    assert(np.isclose(stat, assert_val))


def test_dsr():
    '''
    Example from The Deflated Sharpe Ratio
    '''
    assert_val = 0.90039683444939034

    var_sr_annual = .5
    N = 100
    T = 1250
    skew = -3
    kurt = 10

    sharpe_annual = 2.5

    # convert to daily
    var_sr = var_sr_annual / 250
    sharpe_daily = 2.5 / np.sqrt(250)

    stat = pbo.dsr(sharpe_daily,
                   sharpe_std=np.sqrt(var_sr),
                   N=N, T=T, skew=skew, kurtosis=kurt)

    assert np.isclose(stat, assert_val)


def test_minTRL():
    '''
    Based on Figure 8 of The Sharpe Ratio Efficient Frontier.
    '''
    assert_daily = 2.7311878017281761
    assert_weekly = 2.8288335869452368
    assert_monthly = 27.352920196040301

    # e.g. annualised IS sharpe 2.0, true sharpe 1.0
    # normal IID returns, 95% confidence.
    stats1 = pbo.minTRL(sharpe=2 / np.sqrt(250),
                        skew=0, kurtosis=3,
                        target_sharpe=1 / np.sqrt(250),
                        prob=.95) / 250

    # weekly
    stats2 = pbo.minTRL(sharpe=2 / np.sqrt(52),
                        skew=0, kurtosis=3,
                        target_sharpe=1 / np.sqrt(52),
                        prob=.95) / 52

    # monthly, non-normal
    stats3 = pbo.minTRL(sharpe=3 / np.sqrt(12),
                        skew=-.72, kurtosis=5.78,
                        target_sharpe=2.5 / np.sqrt(12),
                        prob=.95) / 12

    assert(np.isclose(stats1, assert_daily))
    assert(np.isclose(stats2, assert_weekly))
    assert(np.isclose(stats3, assert_monthly))


def test_minBTL():
    '''
    Example in Figure 2 in Computing the Probability of Overfitting in the
    Backtesting and Optimization of Investment Strategies.

    http://www.finanzaonline.com/forum/attachments/econometria-e-modelli-di-trading-operativo/1782818d1377073013-ts-amico-3-shut-up-drive-computing-probability-over-fitting-backtesting-optimization-investment-stra.pdf
    '''
    assert_val = 4.9980870044434038
    min_years, _ = pbo.minBTL(N=45, sharpe_IS=1)
    assert(np.isclose(min_years, assert_val))
