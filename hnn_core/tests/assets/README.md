# Test Assets

This directory contains test assets used by the test suite. These files were previously
downloaded during test execution, but are now included as assets to improve test reliability
and performance.

## Files

| Filename | Size | Original Source | Description |
|----------|------|-----------------|-------------|
| `default.param` | 5.6KB | https://raw.githubusercontent.com/hnnsolver/hnn-core/test_data/default.param | Default parameters file in legacy format |
| `base.json` | 6.4KB | https://raw.githubusercontent.com/jonescompneurolab/hnn-core/test_data/base.json | Base parameters file in JSON format |
| `ERPYes100Trials.param` | 5.7KB | https://raw.githubusercontent.com/jonescompneurolab/hnn/master/param/ERPYes100Trials.param | ERP simulation parameters with drives that should be removed |
| `yes_trial_S1_ERP_all_avg.txt` | 5.1KB | https://raw.githubusercontent.com/jonescompneurolab/hnn/master/data/MEG_detection_data/yes_trial_S1_ERP_all_avg.txt | Data file used for RMSE calculation test |
| `dpl.txt` | 208.1KB | https://raw.githubusercontent.com/jonescompneurolab/hnn-core/test_data/dpl.txt | Dipole data file used for comparing backends |

## Updating

If any of these files need to be updated in the future, you can use the
`download_test_assets.py` script in the root directory to re-download the latest
versions.
