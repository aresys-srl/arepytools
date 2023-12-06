# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
I/O management package
"""
from .channel_iteration import iter_channels
from .io_support import (
    create_new_metadata,
    read_metadata,
    read_raster,
    read_raster_with_raster_info,
    write_metadata,
    write_raster,
    write_raster_with_raster_info,
)
from .point_target_binary import (
    PointSetProduct,
    convert_array_to_point_target_structure,
)
from .point_target_file import read_point_targets_file, write_point_targets_file
from .productfolder2 import create_product_folder, open_product_folder


class ChannelDeprecationWarning(Warning):
    """Custom deprecation warning for the Channel class"""


class ProductFolderDeprecationWarning(Warning):
    """Custom deprecation warning for the ProductFolder class"""


class RenameProductFolderDeprecationWarning(ProductFolderDeprecationWarning):
    """Custom deprecation warning for the rename_product_folder function"""


class RemoveProductFolderDeprecationWarning(ProductFolderDeprecationWarning):
    """Custom deprecation warning for the remove_product_folder function"""
