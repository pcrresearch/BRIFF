# BRIFF
Implementation of BRIFF & FLAMES algorithms, for Fast & Global Point Cloud rigid registration.

## Installation

```
pip install briff
```


## Basic Usage


- **Pairwise**:

```python
from briff import PairwiseBRIFF, icp

briff = PairwiseBRIFF(icp)
T_s2m_hat = briff(sources, models)
# or
registered_sources, registered_models = briff.register(sources, models)
```


- **Generative multiview**:

```python
from briff import GenerativeMultiviewsBRIFF, ChamferJRMPC

briff = GenerativeMultiviewsBRIFF(ChamferJRMPC())
briff(views)
T_hat = briff.T_hat
```