.. _User guide:

IO Module: user guide
=====================


Overview
--------

The purpose of the newly designed IO module, introduced in v1.6.0, is to allow the interaction with Aresys product folder
in an intuitive and fast way, trying to reduce boilerplate code and simplify operations. Several common use cases,
such as looping over channels and reading/writing metadata and raster files, could definitely benefit from the newly
designed utilities.

Of course, in order to simplify the user experience, these new features required a new approach to the design of the IO module,
leading to the discontinuity of older implementations: **ProductFolder** and **Channel** classes above all.

These options have been deprecated since *version 1.6.0* but are still available for legacy purposes. That being said,
they are not going to be maintained in the future, except for specific criticalities and bugs.
The new IO module must be used when writing new code and, old code should be replaced too.

In this user guide there is an overview of the whole IO module organized by theme with useful code snippets that show the
potential of this new approach to internal products management.


Module structure and main features
----------------------------------

The following diagram describes the most relevant files of the IO module.

.. code-block:: text

    arepytools.io
    ├── channel_iteration.py
    ├── io_support.py
    ├── metadata.py
    ├── point_target_binary.py
    ├── point_target_file.py
    └── productfolder2.py
 

The most important and common functionalities, such as opening a Product Folder, reading a point target product,
creating a new metadata or reading a raster file, are **all accessible** just directly in arepytools.io by typing
``import arepytools.io`` without the need of diving too deep inside each sub-module.

For advanced users, here there is a minimal description of each sub-module:

* *channel_iteration*: let the user easily iterate over channels in a Product Folder, returning the channel index and
  the metadata associated to each channel. A filtering function can be applied to select only channels matching the input
  condition, if needed. A custom SwathID (polarization + swath name) filter is available.
* *io_support*: it contains most of the utilities needed to access metadata and raster files on disk: reading/writing functions,
  creation of new metadata objects, row prefix and header (bytes) reading and writing.
* *metadata*: module containing MetaData classes to easily manage .xml channel metadata files. Self-explanatory methods and
  attributes have been implemented to help the user with retrieving the correct MetaDataElements.
* *point_target_binary*: one of the two modules dedicated to managing Aresys Point Target files, namely Point Target
  Binary folders also known as Point Set Products. Both reading and writing features are available.
* *point_target_file*: module dedicated to managing Aresys Point Target .XML files with reading and writing utilities.
* *productfolder2*: this sub-module has been developed to replace the old ``ProductFolder`` class and establish a new and
  improved workflow with easier interfaces and straightforward functions that bring the user experience to a much more understandable
  and clean level.

New Product Folder management workflow
--------------------------------------

To overcome all the limitations of the previous implementation of the ProductFolder object in Python, metadata-raster coupling
above all, a new simpler class called ``ProductFolder2`` has been developed. This class should not be instantiated by the user,
contrary to what was done so far: it is automatically generated as output of ``create_product_folder()``
and ``open_product_folder()`` utilities.

Once this object is created, it can be easily managed without worrying about its open mode state (no more “reading/writing” modes).
Also, it can be deleted using its own method ``.delete()`` or renamed/moved with ``.rename()``.

Channel management: retrieving items location
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Channels management has been rather improved: the list of available channels inside the Product Folder on disk can be accessed
via ``.get_channels_list()`` method of ``ProductFolder2`` object. This list contains the channel identifiers and not the channel
incremental indexes. For example, if a PF contains channel *0001* and channel *0016*, this method returns [1, 16].

This behavior is respectful of the channel real filename and it has been adopted in order to query the exact position on the disk
of specific channel data by requesting this information starting from channel number.

Few querying methods have been exposed to retrieve the full paths on disk of metadata .xml and raster files, .kmz overlay
and config files. The following snippet shows how it works:

.. code-block:: python

	from arepytools.io import open_product_folder
	path_to_product = ...
	product = open_product_folder(path_to_product)
	list_of_channels = product.get_channels_list()
	# for examples [1, 16]
	
	# this will return the absolute path of channel 16 metadata xml file on disk
	ch1_metadata_path = product.get_channel_metadata(16)
	# this will return the absolute path of channel 16 raster file on disk
	ch1_raster_path = product.get_channel_data(16)
	# this will return the absolute path of the .kmz archive
	# (no channel number is needed here, there is only 1 file)
	kmz_file = product.get_overlay_file()

The path can in principle be obtained for any possible channel number: ``.get_channel_metadata(150)`` just returns the full
path of channel 150 even if it does not exist. This behavior is essential when writing new channels data and metadata to
the product folder without any constraint due to the code implementation itself.

Iterating over existing channels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A new sub-module named *channel_iteration* has been developed to simplify the user experience when iterating over all
channels in a Product Folder. The main function called ``iter_channels()`` create a python generator yielding
the channel identifier and the associated metadata object. This improves code readability and performances by lazy
evaluating channel data only on request.

This function also implements by default a *SwathID* filter to select only those channels matching a given polarization (or
multiple polarizations) and/or swath name when provided.

.. code-block:: python

	from arepytools.io import open_product_folder, iter_channels
	path_to_product = ...
	product = open_product_folder(path_to_product)
	
	# without filtering channels
	for channel_id, channel_metadata in iter_channels(product):
		...
	# filtering channels by polarization
	for channel_id, channel_metadata in iter_channels(product, polarization='H/H'):
		...
	# filtering channels by multiple polarizations
	for channel_id, channel_metadata in iter_channels(product, polarization=['H/H', 'V/V']):
		...
	
	# filtering channels by swath id 
	for channel_id, channel_metadata in iter_channels(product, swath='S1'):
		...
    
	# polarization can be specified also using an Enum type
	from arepytools.io.metadata import EPolarization
	# filtering channels by polarization and swath
	for ch_id, ch_metadata in iter_channels(product, polarization=EPolarization.hh, swath='S1'):
		...

``iter_channels()`` can be imported directly from ``arepytools.io`` and it's a wrapper on ``iter_channels_generator()`` that
is a lower level function that takes as an optional argument a filtering function specified by the user. This means that
a user-developed filter can be used to exploit the channel iteration feature customizing the filtering options as needed
for a more advanced usage.

**NB:** while with the old product folder management channels were indexed by their positions in a list of channels,
now channels are indexed by a unique identifier related to the real channel name itself, in this case being the channel number.
This means that all the channel request involving indexes must now be addressed considering the following relationship:
``channel_id = channel_list[channel_index]``.

Loading channel metadata and raster
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For every existing channel, metadata and raster data can be manually loaded using the proper reading functions available
in the ``arepytools.io`` module. Raster files can be easily read providing the correct *RasterInfo* object to the function
``read_raster_with_raster_info()``.
The following code example shows the right procedure to access data and metadata for a given channel.

.. code-block:: python

	from arepytools.io import open_product_folder, read_metadata, read_raster_with_raster_info, iter_channels
	path_to_product = ...
	product = open_product_folder(path_to_product)
	
	list_of_channels = product.get_channels_list()	# for examples [1, 2, 3]
	
	# reading channel 3 metadata
	ch3_metadata = read_metadata(product.get_channel_metadata(3))
	
	# reading channel 3 raster
	ch3_raster = read_raster_with_raster_info(
		raster_file=product.get_channel_data(3),
		raster_info=ch3_metadata.get_raster_info()
	)
	
	# otherwise, using iterchannels
	for channel_num, metadata in iter_channels(product):
		# metadata file is already available
		ch_raster = read_raster_with_raster_info(
			raster_file=product.get_channel_data(channel_num),
			raster_info=metadata.get_raster_info()
		)

Writing channel metadata and raster
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A similar procedure can be followed to write new data and metadata for a given channel inside the Product Folder.
``write_metadata()`` and ``write_raster_with_raster_info()`` utilities are available and their usage is pretty straightforward:

.. code-block:: python

	from arepytools.io import open_product_folder, read_metadata, write_metadata, write_raster_with_raster_info
	from arepytools.io.metadata import EPolarization
	path_to_product = ...
	product = open_product_folder(path_to_product)
	
	list_of_channels = product.get_channels_list()	# for examples [1, 2, 3]
	
	# reading channel 3 metadata
	ch3_metadata = read_metadata(product.get_channel_metadata(3))
	
	# copying channel 3 metadata and writing them as channel 9 metadata
	# changing filename to match the new raster name
	ch3_metadata.filename = product.get_channel_metadata(9).name
	# polarization or other raster info fields can be changed with ease
	ch3_metadata.polarization = EPolarization.vv
	write_metadata(metadata_obj=ch3_metadata, metadata_file=product.get_channel_metadata(9))
	
	# same process but for raster file
	ch3_raster = read_raster_with_raster_info(
		raster_file=product.get_channel_data(3),
		raster_info=ch3_metadata.get_raster_info()
	)
	write_raster_with_raster_info(
		raster_file=product.get_channel_data(9),
		data=ch3_raster,
		raster_info=ch3_metadata.get_raster_info()
	)

``create_new_metadata()`` is a new functionality added to simplify the process of creating a new empty metadata object and
fill it with all the desired information.

.. code-block:: python

	from arepytools.io import create_new_metadata, write_metadata
	import arepytools.io.metadata as mtd
	
	path_to_new_metadata = ...
	new_metadata = create_new_metadata(num_metadata_channels=1, description="new metadata test")
	# creating new metadata information
	raster_info = mtd.RasterInfo(
		lines=8951,
		samples=2215,
		celltype="FLOAT32",
		filename="GRD_0001",
		header_offset_bytes=150,
		row_prefix_bytes=20,
		byteorder="LITTLEENDIAN",
		invalid_value=None,
		format_type=None,
	)
	swath_info = mtd.SwathInfo(...)
	dataset_info = mtd.DataSetInfo(...)
	
	# adding information to metadata object
	new_metadata.insert_element(raster_info)
	new_metadata.insert_element(swath_info)
	new_metadata.insert_element(dataset_info)
	
	# write metadata to disk
	write_metadata(metadata_obj=new_metadata, metadata_file=path_to_new_metadata)


Reading and writing Point Target data
-------------------------------------

Point Target management utilities have been added to let the user read, use, edit and write both Point Target .xml files
and Point Target Binary folders (a.k.a. Point Set Products). Due to the quite different nature of the two Point Target products,
their utilities differ in form and usage.

The following code example shows how to read Point Target products based on their nature.

.. code-block:: python

	from arepytools.io import read_point_targets_file, PointSetProduct
	
	path_to_binary_folder = ...
	path_to_xml_file = ...
	
	# reading Point Target Binary product (a starting reading point and the total length
	# of bytes to be read can be specified)
	point_target_manager = PointSetProduct(
		path=path_to_binary_folder,
		mode="r"
	)
	# output: coordinates array (N, 3), rcs array (N, 4) (HH, HV, VV, VH)
	# data can easily be read by blocks by specifying a start reading byte and the number of
	# bytes to be read
	coordinates_array, rcs_array = point_target_manager.read_data(start=0, num_points=None)
	
	# reading Point Target xml file
	# output: point_targets is a dictionary withe keys being the target id and
	# values being the NominalPointTarget dataclass with info regarding that point target,
	# such as coordinates location and rcs and delay values
	point_targets = read_point_targets_file(path_to_xml_file)


The output results are quite different because the Point Target Binary format can in theory be used to store data for a
very large number of point targets and loading these data and converting information to a dictionary of dataclasses can
be an issue and a performance bottleneck. Therefore, data read using the ``PointSetProduct`` class are returned as numpy arrays,
one with coordinates and the other with RCS values. Nevertheless, the user can easily transform this output to the same
format generated by ``read_point_targets_file()`` using the utility function ``convert_array_to_point_target_structure()``
available in the ``point_target_binary`` sub-module.


Compatibility with legacy code (managing deprecation warning)
-------------------------------------------------------------

Compatibility with legacy code written with the old workflow is still granted although ``productfolder`` and ``channel``
sub-modules have been **officially deprecated**. This means that importing those modules or using functions and classes
defined inside them raises specific warnings suggesting the use of the related new implemented feature.

Custom warnings have been defined to simplify the filtering in older code to avoid displaying warning messages all over.
The two main custom warnings added are: ``ChannelDeprecationWarning`` and ``ProductFolderDeprecationWarning``.
To avoid displaying these warnings, just import the Python warning module and the custom warning to be filtered out and
place the following line of code just before these deprecated modules are imported, as shown in the following example:

.. code-block:: python

	import warnings
	from arepytools.io import ChannelDeprecationWarning
	warnings.filterwarnings("ignore", category=ChannelDeprecationWarning)
	from arepytools.io.channel import Channel


Some examples of old code improved with new features
----------------------------------------------------

The following side-by-side view shows the complexity of overwriting a Product Folder's channels data. With the old workflow,
the original Product Folder was opened in "read mode", its data read, copied and edited if needed but they could not be
overwritten to disk because the product was in "read mode" only: the current product was to be deleted and a new one opened
in "write mode" to be able to write data to disk.
This is now much simpler and pretty straightforward, as you can see in the code example on the right.

+---------------------------------+----------------------------------+
|                                 |                                  |
|.. literalinclude:: old_code.txt |.. literalinclude:: new_code.txt  |
|                                 |                                  |
+---------------------------------+----------------------------------+


Old code vs New code comparison table
-------------------------------------

+------------------------------------------------------------+---------------------------------------------------------------------+
| Old Code                                                   | New code                                                            |
+============================================================+=====================================================================+
|``ProductFolder(path, "r")``                                |``open_product_folder(path)``                                        |
+------------------------------------------------------------+---------------------------------------------------------------------+
|``ProductFolder(path, "w")``                                |``create_product_folder(path)``                                      |
+------------------------------------------------------------+---------------------------------------------------------------------+
|``remove_product_folder(path)``                             |``product.delete()``                                                 |
+------------------------------------------------------------+---------------------------------------------------------------------+
|``rename_product_folder(input_path, new_path)``             |``product.rename(new_path)``                                         |
+------------------------------------------------------------+---------------------------------------------------------------------+
|``for ch_index in range(pf.get_number_channels()):``        |``for ch_id, metadata in iter_channels(pf):``                        |
+------------------------------------------------------------+---------------------------------------------------------------------+
|``pf.get_channel(ch_idx).get_raster_info()``                |``read_metadata(pf.get_channel_metadata(ch_id)).get_raster_info()``  |
+------------------------------------------------------------+---------------------------------------------------------------------+
|``product.write_metadata(ch_idx)``                          |``write_metadata(metadata_object, file)``                            |
+------------------------------------------------------------+---------------------------------------------------------------------+
|``product.get_channel(ch_idx).read_data(block)``            |``read_raster_with_raster_info(file, raster_info, block)``           |
+------------------------------------------------------------+---------------------------------------------------------------------+
|``product.write_data(ch_idx, data)``                        |``write_raster_with_raster_info(file, data, raster_info)``           |
+------------------------------------------------------------+---------------------------------------------------------------------+