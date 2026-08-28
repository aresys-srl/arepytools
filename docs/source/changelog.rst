Changelog
=========

v1.7.7
------

**New features**

- ``insert_element`` method of ``MetaDataChannel`` class now supports an ``overwrite_ok`` option.
- ``metadata.DataSetInfo`` now supports projection parameters and instrument configuration ID.
- direct geocoding initialization takes into account geocoding altitude.

v1.7.6
------

**New features**

- Added support to new optional ImageQuantity tag in DataSetInfo section of the product metadata.

**Bug fixes**

- Fixed bug in inverse geocoding initialization

**Other changes**

- Conda package extension updated to .conda

v1.7.5
------

**Other changes**

- Improved type hint of RasterInfo class


v1.7.4
------

**Bug fixes**

- Fix bug in ``create_orbit``

**Other Changes**

- Code base reformatted with ``ruff``
- Code clean based on ``ruff`` linting 

v1.7.3
------

**New features**

- Added ``get_geometric_doppler_centroid`` function to geometrically compute doppler centroid from squint angle

v1.7.2
------

**Bug fixes**

- Fixed raster data inversion between VH and VV for Point Target Binary reader

v1.7.1
------

**New features**

- IO module, added support to ``INT8_COMPLEX``, ``INT16_COMPLEX`` and ``INT_COMPLEX`` data raster

**Bug fixes**

- Fixed rare rounding issue in `PreciseDateTime`

v1.7.0
------

**Incompatible changes**

- Dropped support to Python 3.8

**New features**

- Added a new Cubic Spline interpolated ``Orbit`` object to substitute the `GeneralSarOrbit`
- Added a dedicated ANX time module
- Added a new `geometry.attitude_utils` module with functionalities to compute yaw, pitch and roll from antenna reference frame and vice-versa
- Added a ``compute_euler_angles_from_rotation`` function as inverse operation for ``compute_rotation`` in `geometry.rotation` module

**Bug fixes**

- Fixed wrong unit of measure in DopplerRate polynomial

**Other Changes**

- Module `arepytools.constants` is now deprecated
- Private module `arepytools._utils` was removed
- ``RasterInfo`` metadata class now has a setter method for the filename field
- Added support for *AzimuthSteeringAngleReferenceTime / AzimuthSteeringAnglePol* new metadata nodes of the internal product format
- Documentation updated with new guide on how to use new Orbit and ANX time module

v1.6.3
------

**Bug fixes**

- Fixed bug when loading ProductFolder manifest path on Linux via a `PosixPath`

v1.6.2
------

**Bug fixes**

- Fix: added missing unit of measures for higher orders in metadata polynomials

v1.6.1
------

**New features**

- Timing sub-package: added `date_to_gps_week` function to convert `PreciseDateTime` or `datetime` objects to GPS weeks 

**Bug fixes**

- Fix: fix an error in `io.read_raster` that could cause integer overflow if image is greater than 2.2 GB
- Fix: fix an error in `GSO3DCurveWrapper` methods where input variable names were not matching its protocol's
- Fix: fix an error in `Generic3DCurve` methods where input variable names were not matching its protocol's


v1.6.0
------

**New features**

- IO sub-package: added `channel_iteration` to assist channel iterations procedures with optional polarization and/or swath filtering
- IO sub-package: added functions to read and write both `Point Target` xml files and binary products
- IO sub-package: private _utils functionalities made public in `io_support`. New functions added and code quality/readability improved
- IO sub-package: added `io_support` with utilities to read and write raster and metadata files
- IO sub-package: added `ProductFolder2` new Product Folder class to safely manage Aresys products
- IO sub-package: added `Product Folder Layout` class to safely manage paths inside `ProductFolder2` class
- IO sub-package: `metadata` module and functions improved and refactored
- Look angles from trajectory: added function to perform look angles computation from a trajectory curve compliant with
  the `TwiceDifferentiable3DCurve` protocol
- Incidence angles from trajectory: added function to perform incidence angles computation from a trajectory curve
  compliant with the `TwiceDifferentiable3DCurve` protocol

**Other Changes**

- IO sub-package: `productfolder` module is now deprecated
- IO sub-package: `channel` module is now deprecated
- Documentation: added a user manual describing the new features and new Product Folder management workflow :ref:`User guide`


v1.5.1
------

**Bug fixes**

- Fix: fix regression on missing description field in manifest file
- Fix: fix regression shape of `general_sar_orbit.sat2earth()` output

**Other Changes**

- IO module: Manifest management updated to version 2.1, removed default description


v1.5.0
------

**Incompatible changes**

- Dropped support to Python 3.7

**New features**

- Direct geocoding attitude: added function to perform direct geocoding taking into account sensor attitude
- Inverse geocoding attitude: added function to perform inverse geocoding taking into account sensor attitude
- IO sub-package: added product folder manifest utilities (`arepytools.io.manifest`)
- Geometry: added geometric squint evaluation to `arepytools.geometry.geometric_functions`

**Bug fixes**

- Fix: fixed duplicated lines in XML parsing

**Other Changes**

- Geometry: inverse geocoding methods vectorized
- Geometry: direct geocoding methods vectorized
- Private `geometry._geocoding` sub-package removed


v1.4.0
------

**New features**

- IO sub-package: added metadata support to PRF changes and chirp period in the `AcquisitionTimeLine`;

**Bug fixes**

- Fix: fix wrong annotated units of measure in all polynomials of the metadata (e.g. `DopplerRate`, `SlantToGround`);

**Other Changes**

- IO sub-package: external `pyxb` dependency replaced with `xsdata`;
- Performance improvement in metadata input/output operations: several times faster on test data, results varying with metadata size and hardware;

v1.3.1
------

**Incompatible changes**

- Class :code:`arepytools.io.channel.Channel`: initialization with invalid raster file name is no longer supported;

**Bug Fixes**

- Fix: Fix support to product folder whose name contains dots;

v1.3.0
------

**New features**

- Added flag in :code:`generalsarorbit.create_general_sar_orbit` to ignore invalid ANX time

**Incompatible changes**

- Class member :code:`math.genericpoly.GenericPoly.poly` is no longer a dict{key:value} but is a list[(key, value)];

**Bug Fixes**

- Fix: bistatic inverse geocoding proper initialization added to avoid unphysical solutions;
- Fix: :code:`math.genericpoly.GenericPoly` init fixed to support to coefficients with same values;
- Fix: Unexpected zero channel issue with input path ending with '/', fixed in ProductFolder;
- Fix: Fix support to product folder with tiff raster;
- Fix: Added missing 'X/X' polarization to :code:`SwathInfo` and :code:`AntennaInfo` metadata;
- Fix: Corrected :code:`open mode` type hint in :code:`ProductFolder` and :code:`Channel`;
- Fix: Support extended to numpy 1.24 in :code:`_direct_geocoding_monostatic_init`;

v1.2.0
------

**New features**

- Added the methods :code:`geometry.generalsarattitude.compute_antenna_reference_frame`, :code:`GeneralSarAttitude.get_arf`,
  :code:`geometry.generalsarattitude.compute_pointing_directions` and :code:`GeneralSarAttitude.sat2earthLOS`;
- Added the :code:`reference_frames` and :code:`rotation` modules;
- Added the method :code:`geometry.ellipsoid.compute_line_ellipsoid_intersections` and :code:`Ellispoid.inflate`;
- Added a public module :code:`directgeocoding` with :code:`direct_geocoding_with_looking_direction` and :code:`direct_geocoding_with_look_angles`;
- Added the geometry module :code:`geometric_functions`, with :code:`compute_incidence_angles` and :code:`compute_look_angles`;
- Added the methods :code:`generalsarorbit.compute_look_angles_from_orbit` and :code:`generalsarorbit.compute_incidence_angles_from_orbit`
  and :code:`generalsarorbit.compute_ground_velocity`;
- Added the methods :code:`PreciseDateTime.now`, :code:`PreciseDateTime.from_sec85`, :code:`PreciseDateTime.from_utc_str`, 
  :code:`PreciseDateTime.from_numeric_datetime` and :code:`PreciseDateTime.fromisoformat`;

**Incompatible changes**

- Enum :code:`arepytools.geometry.generalsarattitude.Angles` was removed as not necessary;
- Generator :code:`arepytools.geometry.generalsarattitude.EnumerateRot` was removed as not necessary.

**Other Changes**

- The :code:`_ellipsoid` module is now public: :code:`ellipsoid`;
- The module :code:`arepytools.geometry.wgs84` is deprecated, its content is available in :code:`arepytools.geometry.ellipsoid`;
- The :code:`PreciseDateTime().set_now()` methods is deprecated, use :code:`PreciseDateTime.now()`;
- The :code:`PreciseDateTime().set_from_sec85(...)` methods is deprecated, use :code:`PreciseDateTime.from_sec85(...)`;
- The :code:`PreciseDateTime().set_from_utc_str(...)` methods is deprecated, use :code:`PreciseDateTime.from_utc_str(...)`;
- The :code:`PreciseDateTime().set_from_numeric_datetime(...)` methods is deprecated, use :code:`PreciseDateTime.from_numeric_datetime(...)`;
- The :code:`PreciseDateTime().set_from_isoformat(...)` methods is deprecated, use :code:`PreciseDateTime.fromisoformat(...)`;
- Performance of PreciseDateTime arithmetic operations improved;

v1.1.0
------

**New features**

- Added read/write I/O functionalities to handle metadata attributes (e.g. contentID and description);
- Added the method :code:`MetaDataChannel.remove_element`;
- Added the method :code:`io.productfolder.get_channel_indexes`;
- Added the method :code:`io.productfolder.rename_product_folder`;
- Added the method :code:`io.productfolder.remove_product_folder`;
- Added :code:`config_file` property to :code:`io.productfolder.ProductFolder`;
- Added :code:`raster_file` and :code:`metadata_file` properties to :code:`io.channel.Channel`;
- Function to compute the ANX times for a given orbit added to generalsarorbit module (:code:`compute_anx_times`);
- Function to compute the number of ANX for a given orbit added to generalsarorbit module (:code:`compute_number_of_anx`);
- Added the method :code:`GeneralSarOrbit.get_time_since_anx`;
- Added the properties :code:`GeneralSarOrbit.anx_times` and :code:`GeneralSarOrbit.anx_positions`;
- Added support for ANX to :code:`GeneralSarOrbit` constructor and :code:`create_general_sar_orbit` function;
- ANX information management added in state vectors metadata conversion functions.

**Bug Fixes**

- Fix: :code:`BurstInfo.get_burst_roi` wrong index in the roi_range;
- Fix: :code:`MetaDataChannel.get_supported_metadata_elements` removed duplicated Burstinfo in supported_elements;
- Checks on ANX time and position in :code:`StateVectors` class fixed.

**Other changes**

- I/O module: autocompletion improved (type hints), exceptions cleaned and documentation updated;
- I/O module: :code:`file_name` property of :code:`io.channel.Channel` deprecated;
- :code:`StateVectors.get_anx_time` method deprecated in favor of new :code:`StateVectors.anx_time` property;
- :code:`StateVectors.set_axn_info` method deprecated in favor of new :code:`StateVectors.set_anx_info` method;
- Drop support for Python 3.5 and 3.6.

v1.0.1
------

**Bug Fixes**

- Fix: :code:`GeneralSarOrbit.earth2sat` wrong getter name used.

**Other changes**

- Autocompletion of I/O module improved (type hints) and documentation updated.

v1.0.0
------

**New features**

- Added debug I/O functions to :code:`arepytools.io` package;
- :code:`Channel` and :code:`ProductFolder` classes can be used to read and
  write binary header and row prefix, added methods :code:`read_binary_header`,
  :code:`write_binary_header`, :code:`read_row_prefix` and
  :code:`write_row_prefix`;
- :code:`PreciseDateTime` class supports ISO 8601 strings, added methods
  :code:`fromisoformat`, :code:`isoformat` and :code:`set_from_isoformat`;
- Added data file extension support to :code:`ProductFolder` (read only mode),
  GeoTiff are now supported;
- Added conversion methods from :code:`_Poly2D`/:code:`_Poly2DVector` to
  :code:`GenericPoly`/:code:`SortedPolyList`.

**Bug Fixes**

- Fix :code:`GeneralSarOrbit.sat2earh` input checks;
- Fix :code:`PreciseDateTime.__repr__` to ignore current locale configuration;
- Fix :code:`write_metadata` function, for writing missing attributes;
- Fix :code:`write_raster` function;
- Fix inverse geocoding initialization;
- Fix channel delay value retrieval in SwathInfo metadata conversion function.

**Incompatible changes**

- Removed :code:`check_args_are_numeric` in submodule :code:`arepytools._utils`;
- :code:`read_raster` and :code:`write_raster` functions parameters changed.

**Other changes**

- :code:`ProductFolder` robustness improved;
- :code:`read_raster` and :code:`write_raster` robustness and performance
  improved;
- :code:`GeometryInterpolator` performance improved;
- Removed legacy convergence check in Newton solver;
- Documentation updated.

v1.0.0b2
--------
Second beta version.

**Bug Fixes**

- Fix is_productfolder function: an empty product folder is a product folder;
- Fix channel getter of SlantToGround and GroundToSlant;
- Fix bistatic inverse geocoding;
- Fix unmarshalling of OrbitNumber and Track properties of state vector data;
- Fix unmarshalling of Swl_changes_number property of acquisition timeline.

**Incompatible changes**

- Append and overwrite ProductFolder open modes currently disabled;
- Removed unused show_waitbar parameter from data read/write functions;
- Complex int16 and int8 currently marked as unsupported in data read/write functions;
- read_raster function no longer performs data transposition;
- wgs84 object renamed to WGS84.

**Other changes**

- Documentation updated.

v1.0.0b1
--------
First beta version.
