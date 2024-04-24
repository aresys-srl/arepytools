Changelog
=========

v1.6.2
------

First official release.

**Bug fixes**

- Fixed bug when loading ProductFolder manifest path on Linux via a `PosixPath`
- Fixed wrong unit of measure in DopplerRate polynomial
- Fix: added missing unit of measures for higher orders in metadata polynomials

**Other Changes**

- Module `arepytools.constants` is now deprecated.
- Private module `arepytools._utils` was removed.

v1.6.1
------

**New features**

- Timing sub-package: added `date_to_gps_week` function to convert `PreciseDateTime` or `datetime` objects to GPS weeks 

**Bug fixes**

- Fix: fix an error in `io.read_raster` that could cause integer overflow if image is greater than 2.2 GB
- Fix: fix an error in `GSO3DCurveWrapper` methods where input variable names were not matching its protocol's
- Fix: fix an error in `Generic3DCurve` methods where input variable names were not matching its protocol's
