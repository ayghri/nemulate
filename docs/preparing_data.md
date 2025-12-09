

To download the CESM2 data, refer to the guide from "bulkget" package:
   https://github.com/ayghri/bulkget/blob/master/examples/cesm_download.md


Once that done, you should have a folder structure of 
`{VARIABLE}/{MEMBER_ID}/{START MONTH}-{END MONTH}.nc`.

We now going to change that to `{VARIABLE}/{MEMBER_ID}.nc` via `scripts/00_prepare_data.py`, then use time dimension chunks when we want to load data. Use `python  script_path.py --help`
to see how the script should be used. It's straight forward. When in doubt, add `--dry-run` flag for a mock call.

** The rest of the package assumes that data is stored in the merged format `{VARIABLE}/{MEMBER_ID}.nc`**

## Compute forced moments
The next step it to compute the forced response (first moment) and *forced* second moment that is needed to compute the mean and standard deviation for different variables, as well as to train on internal variability if needed.
This is handled by `script/01_compute_forced_moments.py`.



