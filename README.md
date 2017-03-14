# pypbo
Probability of Backtest Overfitting in Python

Python implmenetation of Probability of Backtest Overfitting. [1]

## Features:

* Probability of backtest overfitting
* Probability of Out of Sample (OOS) Below Threshold
* Performance Degradation
* Stochastic Dominance
* Probabilistic Sharpe Ratio (PSR) statistics [2]
* Minimum track record length (MinTRL) [2]
* Minimum backtest length (MinBTL) [3]
* Deflated Sharpe Ratio statistics [4]

## TODO:

* Add test cases.
* Optimial N trials [4]
* Harvey and Liu 2014 paper on sharpe ratio threshold. [5]


## Required Packages:

* `joblib`
* `seaborn`
* `statsmodels 0.8.0`

## Usage:

```python
import pypbo as pbo
import pypbo.perf as perf

def metric(x):
    return np.sqrt(255) * perf.sharpe_iid(x)

S = 16

pbox = pbo.pbo(rtns_df, S=S,
               metric_func=metric, threshold=1, n_jobs=4,
               plot=True, verbose=False, hist=False)
```


## Testing

Test scripts use `pytest` package. To run tests, as an example, run the
following script. Make sure `py.test` is in your `PATH` variable.

```python
$ > py.test pypbo/tests/pbo_test.py
```

## Reference
---------
[1] Bailey, David H. and Borwein, Jonathan M. and Lopez de Prado, Marcos and Zhu, Qiji Jim, The Probability of Backtest Overfitting (February 27, 2015). Journal of Computational Finance (Risk Journals), 2015, Forthcoming. Available at SSRN: http://ssrn.com/abstract=2326253 or http://dx.doi.org/10.2139/ssrn.2326253

[2] Bailey, David H. and Lopez de Prado, Marcos, The Sharpe Ratio Efficient Frontier (April 2012). Journal of Risk, Vol. 15, No. 2, Winter 2012/13. Available at SSRN: http://ssrn.com/abstract=1821643 or http://dx.doi.org/10.2139/ssrn.1821643

[3] Bailey, David H. and Borwein, Jonathan M. and Lopez de Prado, Marcos and Zhu, Qiji Jim, Pseudo-Mathematics and Financial Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample Performance (April 1, 2014). Notices of the American Mathematical Society, 61(5), May 2014, pp.458-471. Available at SSRN: http://ssrn.com/abstract=2308659 or http://dx.doi.org/10.2139/ssrn.2308659

[4] Bailey, David H. and Lopez de Prado, Marcos, The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality (July 31, 2014). Journal of Portfolio Management, 40 (5), pp. 94-107. 2014 (40th Anniversary Special Issue).. Available at SSRN: http://ssrn.com/abstract=2460551

[5] Harvey, Campbell R. and Liu, Yan, Backtesting (July 28, 2015). Available at SSRN: http://ssrn.com/abstract=2345489 or http://dx.doi.org/10.2139/ssrn.2345489
