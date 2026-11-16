| grid | station [m] | operator | peak [m] | peak err | arrival [s] | err vs interp | err vs sample | verdict |
|---|---|---|---|---|---|---|---|---|
| dx0.04 (dx=0.04) | 2.228 | slab | 0.4003 | +8.2% | 0.4074 | +17.0% | +13.6% | BORDERLINE |
| dx0.04 (dx=0.04) | 2.228 | height | 0.3848 | +4.0% | 0.4358 | +25.2% | +21.4% | BORDERLINE |
| dx0.04 (dx=0.04) | 2.228 | surface | 0.4248 | +14.8% | 0.4018 | +15.4% | +12.5% | FAIL |
| dx0.04 (dx=0.04) | 2.426 | slab | 0.4272 | +15.5% | 0.4604 | +32.3% | +29.2% | FAIL |
| dx0.04 (dx=0.04) | 2.426 | height | 0.4093 | +10.6% | 0.4841 | +39.1% | +35.9% | FAIL |
| dx0.04 (dx=0.04) | 2.426 | surface | 0.4493 | +21.4% | 0.4658 | +33.8% | +30.3% | FAIL |
| dx0.02 (dx=0.02) | 2.228 | slab | 0.3817 | +3.2% | 0.3893 | +11.8% | +9.2% | BORDERLINE |
| dx0.02 (dx=0.02) | 2.228 | height | 0.3989 | +7.8% | 0.3998 | +14.8% | +11.4% | BORDERLINE |
| dx0.02 (dx=0.02) | 2.228 | surface | 0.4189 | +13.2% | 0.3869 | +11.2% | +8.0% | BORDERLINE |
| dx0.02 (dx=0.02) | 2.426 | slab | 0.4767 | +28.8% | 0.4409 | +26.7% | +23.6% | FAIL |
| dx0.02 (dx=0.02) | 2.426 | height | 0.5019 | +35.7% | 0.4541 | +30.4% | +27.0% | FAIL |
| dx0.02 (dx=0.02) | 2.426 | surface | 0.5219 | +41.1% | 0.4356 | +25.1% | +21.4% | FAIL |
| dx0.01 (dx=0.01) | 2.228 | slab | 0.3674 | -0.7% | 0.3869 | +11.1% | +8.5% | BORDERLINE |
| dx0.01 (dx=0.01) | 2.228 | height | 0.3696 | -0.1% | 0.3918 | +12.6% | +9.4% | BORDERLINE |
| dx0.01 (dx=0.01) | 2.228 | surface | 0.3796 | +2.6% | 0.3851 | +10.6% | +7.7% | BORDERLINE |
| dx0.01 (dx=0.01) | 2.426 | slab | 0.4813 | +30.1% | 0.4378 | +25.8% | +22.4% | FAIL |
| dx0.01 (dx=0.01) | 2.426 | height | 0.4773 | +29.0% | 0.4446 | +27.7% | +24.0% | FAIL |
| dx0.01 (dx=0.01) | 2.426 | surface | 0.4873 | +31.7% | 0.4360 | +25.2% | +21.6% | FAIL |

**verdict at the admissible station**: `BORDERLINE` — arrival tolerance is not decidable: +11.1% against the interpolated reference (0.3481 s) but +8.5% against the sample-quantised one (0.3591 s); the reference-definition band is 3.2%

- peak: observed order p=0.38 -> **non-asymptotic**, no extrapolation; finest-two band [0.36741227272727267, 0.38170400000000004]
- arrival: observed order p=2.88, h->0 = 0.386471, GCI_fine=0.12%
- 8 mm station ambiguity (2.220 vs 2.228 m), estimated from the local arrival derivative: +0.54% (worst grid, dx0.02); local front speed 3.84 m/s
- 8 mm station ambiguity, **measured** (dx0.02, aperture 0.08 m): arrival -0.58% (2.220 vs 2.228 m), peak +3.23%
- 8 mm station ambiguity, **measured** (dx0.04, aperture 0.08 m): arrival -0.98% (2.220 vs 2.228 m), peak +0.22%
