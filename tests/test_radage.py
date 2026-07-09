import pytest
from pytest import approx
from radage import radage

def test_rho206238_207235_to_rho207206_238206():
    r206238 = 0.156735819280412
    r206238_std = 0.00124402549361756
    r207235 = 1.61292292028105
    r207235_std = 0.0179718038407983
    rho206238_207235 = 0.954275585240443

    rho2, X2_sig, Y2_sig = \
    radage.rho206238_207235_to_rho207206_238206(
        r206238,
        r206238_std,
        r207235,
        r207235_std,
        rho206238_207235
    )
    assert rho2 == approx(-0.629, abs=0.001)
    assert X2_sig == approx(0.0506, abs=0.001)
    assert Y2_sig == approx(0.0003199, abs=0.0001)
