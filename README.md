# `test_data` branch

This is the branch used for storing some, but not all, of our test data that `hnn-core` regularly runs its test suite against.

Files:

- `base.json`: A "flat JSON" parameter file used for testing loading of params using the legacy methods in `params.py` etc., which rely on a single list of keys whose identifiers are delimited by underscores.
- `default.param`: The "HNN Original" version of parameter files, similar to "flat JSON" files but in a simpler file format.
- `dpl.txt`: The original (scaled and smoothed) "ground truth" dipole data from HNN Original meant for testing that `hnn-core` can reproduce the exact same dipole data. This is **exactly equivalent** to running the following using `hnn-core==0.6.1`:

    ```python
    from hnn_core import neymotin_2020_model, simulate_dipole
    net = neymotin_2020_model(add_drives_from_params=True, legacy_mode=True)
    dpls = simulate_dipole(net, tstop=170.0)
    dpl = dpls[0].smooth(30).scale(3000)
    dpl.write("dpl_reran.txt")
    ```

    After v0.7.x, this dipole data will no longer be used for comparison for a number of reasons, including NEURON 9.0 no longer supporting "legacy" unit/constant values, intended deprecation of the old `params` code in favor of using the API, and other reasons.

- `dpl_nonlegacy_seeds.txt`: This is a new set of "ground truth" dipole data meant for comparison in future testing, which results from a simulation that uses the same parameters as `dpl.txt` *except* that the random seeds used for the evoked drives are slightly different (related to #1340). This will be eventually replaced by another set of "ground truth" dipole data that includes the changes from #1340, #1230, and #1180, all of which will slightly change the dipole output data produced by the default `hnn-core` model (`neymotin_2020_model`) using the "default" ERP drives.
