# Register HVAC-MFR in MMSegmentation

The provided HVAC-MFR configs already include custom_imports, so manual registration is usually not required. If you want permanent registration in an existing MMSegmentation checkout, add the following imports:

```python
# mmseg/models/backbones/__init__.py
from .hvac_mfr import HVACMFR

# mmseg/models/decode_heads/__init__.py
from .hvac_mfr_head import HVACMFRHead
```

If your local MMCV version is newer than the original MMSegmentation bound, adjust `MMCV_MAX` in `mmseg/__init__.py` according to your environment.
