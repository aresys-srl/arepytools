# Arepytools

`Arepytools` is the Aresys Python toolbox for SAR data processing.

This pure python library has been developed through the years to provide accessible essential tools needed to process,
analyze and manage SAR products in the internal Aresys Product Folder format.

This package is composed by several thematic modules containing the most important and useful features related to the
given topic, namely:

- ``io``: input/output module to manage Product Folder format, both metadata and raster files
- ``math``: mathematical assets to help managing polynomials and axes types
- ``timing``: custom times for SAR time coordinates and dates management
- ``geometry``: attitudes and orbits, angles computation functions, coordinates conversions, direct and inverse geocoding functionalities

The package can be installed via pip:

```shell
pip install arepytools
```
