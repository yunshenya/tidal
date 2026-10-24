"""Regression coverage for causal replay metrics."""
import numpy as np


def test_bootstrap_ignores_single_class_draws():
    from tidal.metrics import paired_delta
    y=np.array([0,0,1,1]); p=np.array([.1,.2,.8,.9])
    r=paired_delta(y,p,p,.5,.5,np.array(['a','a','b','b']),n_boot=200)
    assert r['p_le0'] == 1.
    assert 0 < r['n_boot_valid'] < 200
    one=paired_delta(np.zeros(4),p,p,.5,.5,np.array(['a','a','b','b']),n_boot=20)
    assert np.isnan(one['p_le0']) and one['n_boot_valid'] == 0
