# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Product Folder Utils module
---------------------------
"""

import re
from pathlib import Path
from typing import List, Optional, Union

from arepytools.io.manifest import Manifest
from arepytools.io.productfolder_layout import (
    METADATA_EXTENSION,
    ProductFolderLayout,
    QuicklookExtensions,
    RasterExtensions,
)


class PathAlreadyExistsError(RuntimeError):
    """The path provided already exist"""


class ProductFolderNotFoundError(RuntimeError):
    """The Product Folder of choice does not exist"""


class InvalidProductFolder(RuntimeError):
    """The Product Folder of choice is not valid (not a directory or
    does match validity constraints)"""


class ProductFolder2:
    """Product Folder 2 main object"""

    def __init__(
        self,
        path: Union[str, Path],
        raster_extension: Union[str, RasterExtensions] = RasterExtensions.RAW,
    ) -> None:
        """Generate a Product Folder object from its path and the extension of its data files.

        Parameters
        ----------
        path : Union[str, Path]
            path to the selected Product Folder
        raster_extension : Union[str, RasterExtensions], optional
            extension of channel raster data files, by default RasterExtensions.RAW
        """
        self._set_up(path, raster_extension)

    def _set_up(
        self,
        path: Union[str, Path],
        raster_extension: Union[str, RasterExtensions],
    ):
        self._path = Path(path)
        self._layout = ProductFolderLayout(path=self._path)
        self._pf_name = self._layout.product_name
        self._manifest = ProductFolderLayout.generate_manifest_path(self._path)
        self._raster_extension = RasterExtensions(raster_extension)

    def rename(self, new_path: Union[str, Path]):
        """Renaming current product folder with new path.

        Parameters
        ----------
        new_path : Union[str, Path]
            new full path for the current product folder
        """
        rename_product_folder(self._path, new_path)
        self._set_up(
            path=new_path,
            raster_extension=self._raster_extension,
        )

    def delete(self):
        """Deleting current product on disk"""
        delete_product_folder_content(self)

    @property
    def pf_name(self) -> str:
        """Get the Product Folder base name."""
        return self._pf_name

    @property
    def path(self) -> Path:
        """Get the Product Folder path."""
        return self._path

    @property
    def manifest(self) -> Path:
        """Get manifest full path."""
        return self._manifest

    @property
    def raster_extension(self) -> str:
        """Get the channel raster file extension"""
        return self._raster_extension.value

    def get_channels_list(self) -> List[int]:
        """Retrieving list of available channels on disk for the current Product Folder.

        Returns
        -------
        List
            list of integers corresponding to the available channels
        """

        if not is_valid_product_folder(pf_path=self._path):
            raise InvalidProductFolder(self._path)

        files = [f.name for f in self._path.iterdir()]

        headers = [
            f
            for f in files
            if f.startswith(self._path.name) and f.endswith(METADATA_EXTENSION)
        ]
        headers = [h.rstrip(METADATA_EXTENSION) for h in headers]
        channels = [h.split("_")[-1] for h in headers]

        return sorted([int(c) for c in channels])

    def get_channel_metadata(self, channel: int) -> Path:
        """Getter method for retrieving Product Folder channel's metadata path.

        Parameters
        ----------
        channel : int
            channel id

        Returns
        -------
        Path
            Path to the selected channel's metadata files
        """
        return self._layout.get_channel_metadata_path(channel_id=channel)

    def get_channel_data(self, channel: int) -> Path:
        """Getter method for retrieving Product Folder channel's raster file path.

        Parameters
        ----------
        channel : int
            channel id

        Returns
        -------
        Path
            Path to the selected channel's raster file
        """
        return self._layout.get_channel_data_path(
            channel_id=channel, extension=self._raster_extension
        )

    def get_config_file(self) -> Path:
        """Getter method for retrieving Product Folder config file path.

        Returns
        -------
        Path
            Path to the Product Folder config file
        """
        return self._layout.get_config_path()

    def get_overlay_file(self) -> Path:
        """Getter method for retrieving Product Folder overlay (.kmz) file path.

        Returns
        -------
        Path
            Path to the Product Folder overlay file
        """
        return self._layout.get_overlay_path()


def get_channel_quicklook(
    pf: ProductFolder2, ext: Union[str, QuicklookExtensions], channel_id: int
) -> Path:
    """Get quicklook full path for the selected channel.

    Parameters
    ----------
    pf : ProductFolder2
        ProductFolder instance
    ext : Union[str, QuicklookExtensions]
        quicklook extension
    channel_id : int
        channel number

    Returns
    -------
    Path
        Path to the quicklook image corresponding to the selected channel
    """

    ext = QuicklookExtensions(ext)
    return pf._layout.get_channel_quicklook_path(channel_id=channel_id, extension=ext)


def create_product_folder(
    pf_path: Union[str, Path],
    raster_extension: Union[str, RasterExtensions] = RasterExtensions.RAW,
    overwrite_ok: bool = False,
    product_folder_description: Optional[str] = None,
) -> ProductFolder2:
    """Create a new Product Folder layout at the desired location. If the path provided
    already exists, it raises an error unless the overwrite_ok flag is set to True.
    In this case, if the target location corresponds to a valid Product Folder, the Product
    is erased and overwritten.

    Parameters
    ----------
    pf_path : Union[str, Path]
        path to the location where to create the Product Folder
    raster_extension : Union[str, RasterExtensions], optional
        extension of the channel's raster data file to be written in Manifest,
        by default RasterExtensions.RAW
    overwrite_ok : bool, optional
        if True, if a valid Product Folder is located at the given path, it is overwritten,
        by default False
    product_folder_description : str, optional
        product folder description, by default None

    Returns
    -------
    ProductFolder2
        Product Folder object

    Raises
    ------
    InvalidProductFolder
        if path already exist but the path does not correspond to a valid
        Product Folder
    PathAlreadyExistsError
        if input path already exist but overwrite permissions are not given
    """

    pf_path = Path(pf_path)

    # creating ProductFolder2 object and manifest
    product_folder = ProductFolder2(path=pf_path, raster_extension=raster_extension)
    manifest = Manifest(
        description=product_folder_description, datafile_extension=raster_extension
    )

    if pf_path.exists():
        # path already exist but no overwrite permission
        if not overwrite_ok:
            raise PathAlreadyExistsError(
                f"Path {pf_path} already exists. "
                + "To overwrite an existing Product Folder "
                + "set 'overwrite_ok' to True"
            )

        # path already exist, overwrite permission but no valid ProductFolder2
        if not is_product_folder(pf_path):
            raise InvalidProductFolder(
                f"Path {pf_path} does not " + "correspond to a valid Product Folder"
            )

        # path already exist, overwrite permission and valid ProductFolder2
        delete_product_folder_content(product_folder)

    # creating folder and dumping manifest to disk
    pf_path.mkdir(parents=True, exist_ok=True)
    manifest.write(product_folder.manifest)

    return product_folder


def open_product_folder(
    pf_path: Union[str, Path],
) -> ProductFolder2:
    """Open Product Folder at the provided location.

    Parameters
    ----------
    pf_path : Union[str, Path]
        path to the location where to create the Product Folder

    Returns
    -------
    ProductFolder2
        Product Folder object

    Raises
    ------
    ProductFolderNotFoundError
        if path provided does not exist
    InvalidProductFolder
        if path provided exist but does not correspond to a valid Product Folder
    """

    pf_path = Path(pf_path)

    # check if path does not exist
    if not pf_path.exists():
        raise ProductFolderNotFoundError(f"Product Folder {pf_path} does not exists")

    # check if path corresponds to a valid Product Folder
    if not is_valid_product_folder(pf_path):
        raise InvalidProductFolder(
            f"Path {pf_path} does not " + "correspond to a valid Product Folder"
        )

    # reading manifest and creating ProductFolder2 object
    manifest = Manifest.from_file(
        ProductFolderLayout.generate_manifest_path(pf_path=pf_path)
    )
    product_folder = ProductFolder2(
        path=pf_path,
        raster_extension=manifest.datafile_extension,
    )

    return product_folder


def is_product_folder(pf_path: Union[str, Path]) -> bool:
    """Check if input path corresponds to a valid Product Folder, basic version.

    Conditions to be met for Product Folder validity:
        - path exists
        - path is a directory
        - manifest exist
        - manifest can be loaded

    Parameters
    ----------
    pf_path : Union[str, Path]
        path to Product Folder to be checked

    Returns
    -------
    bool
        true if path is a Product Folder, else False
    """
    pf_path = Path(pf_path)
    try:
        Manifest.from_file(ProductFolderLayout.generate_manifest_path(pf_path=pf_path))
    except Exception:
        return False

    return True


def is_valid_product_folder(pf_path: Union[str, Path]) -> bool:
    """Check if the input path corresponds to a valid Product Folder.

    Conditions to be met for Product Folder validity:
        - path exists
        - path is a directory
        - manifest exists
        - pairing between channel metadata and data, aka a raster for each header

    Parameters
    ----------
    pf_path : Union[str, Path]
        path to the selected Product Folder

    Returns
    -------
    bool
        True if path corresponds to a valid Product Folder, else False
    """

    pf_path = Path(pf_path)

    # path does not exist
    if not pf_path.exists():
        return False

    # path is not a folder
    if not pf_path.is_dir():
        return False

    # instantiating theoretical Product Folder directory layout
    layout = ProductFolderLayout(pf_path)

    # determining validity conditions
    manifest_path = layout.generate_manifest_path(pf_path=pf_path)
    try:
        manifest = Manifest.from_file(manifest_path)
    except Exception:
        return False

    return _check_channel_data_pairing_condition(
        pf_path, raster_extension=manifest.datafile_extension
    )


def _check_channel_data_pairing_condition(
    path: Path, raster_extension: RasterExtensions
) -> bool:
    """Determining if the pairing condition between Product Folder channel data
    and metadata is met, aka there is a metadata .xml header file for each raster file.

    Parameters
    ----------
    path : Path
        path to the Product Folder of choice
    raster_extension : RasterExtensions
        raster files extensions to be searched for

    Returns
    -------
    bool
        True if condition is met, else False
    """

    # list files in directory
    files = [f.name for f in path.iterdir()]

    try:
        # detect headers and raster files inside directory
        # searching for .XML files only
        headers_on_disk = [
            f
            for f in files
            if f.startswith(path.name) and f.endswith(METADATA_EXTENSION)
        ]
        # searching for all files in folder without an extension
        raw_raster_on_disk = [
            f
            for f in files
            if f.startswith(path.name) and "." not in f.replace(path.name, "")
        ]
        tiff_raster_on_disk = [
            f
            for f in files
            if f.startswith(path.name) and f.endswith(RasterExtensions.TIFF.value)
        ]

        # found tiff files on disk, but manifest extension is RAW
        if tiff_raster_on_disk and raster_extension == RasterExtensions.RAW:
            return False
        # found RAW files on disk, but manifest extension is TIFF
        if raw_raster_on_disk and raster_extension == RasterExtensions.TIFF:
            return False
        raster_on_disk = (
            tiff_raster_on_disk
            if raster_extension == RasterExtensions.TIFF
            else raw_raster_on_disk
        )

        # check matching between headers filenames and rasters filenames
        headers_clean = [h.rsplit(".", 1)[0] for h in headers_on_disk]
        raster_clean = (
            [r.rsplit(".", 1)[0] for r in raster_on_disk]
            if raster_extension == RasterExtensions.TIFF
            else raster_on_disk.copy()
        )
        headers_clean.sort()
        raster_clean.sort()

        return raster_clean == headers_clean

    except Exception:
        return False


def delete_product_folder_content(product_folder: ProductFolder2) -> None:
    """Safely deleting a Product Folder on disk at the selected location.

    Removing main constituents of a Product Folder based on ProductFolder2 input object.
    If more files than the essential ones are found inside the Product Folder directory,
    the whole directory is not removed and those files only are preserved.
    Quicklook files are not considered part of the ProductFolder structure, so they are always kept.

    If there any files left in the folder after removing the main constituents, the manifest file is formatted to the
    default factory value and the folder is not deleted. Otherwise, if only the manifest file is left, it is removed
    along with the folder itself.

    Parameters
    ----------
    product_folder : ProductFolder2
        Product Folder object corresponding to the Product Folder on disk to be removed
    """

    files = [f.name for f in product_folder.path.iterdir()]
    # removing all files matching the pattern: ...\pf_name_XXXX and ending with .xml, .tiff, and no suffix
    # finding all channels data
    matching_pattern = re.compile(product_folder.pf_name + "_" + r"\d{4}")
    matching_files = [f for f in files if bool(matching_pattern.search(f))]
    # removing channel metadata (.xml)
    metadata = [m for m in matching_files if m.endswith(METADATA_EXTENSION)]
    raster_tiff = [t for t in matching_files if t.endswith(RasterExtensions.TIFF.value)]
    raster_raw = [r for r in matching_files if r[-1].isdigit()]

    files_to_be_deleted = metadata + raster_raw + raster_tiff
    for file in files_to_be_deleted:
        product_folder.path.joinpath(file).unlink()

    # removing config file, if it exists
    product_folder.get_config_file().unlink(missing_ok=True)

    # removing kmz file, if it exists
    product_folder.get_overlay_file().unlink(missing_ok=True)

    # resetting manifest to default: this operation is needed to reset the product folder status so that this folder
    # can be used as a pristine new Product Folder later on, if needed
    default_manifest = Manifest()
    default_manifest.write(product_folder.manifest)

    # however, it the manifest is the only file left in the folder, the whole folder can be deleted from disk
    residual_files = [f.name for f in product_folder.path.iterdir()]
    if len(residual_files) == 1 and residual_files[0] == product_folder.manifest.name:
        product_folder.manifest.unlink()
        product_folder.path.rmdir()


def rename_product_folder(
    current_folder: Union[str, Path], new_folder: Union[str, Path]
) -> None:
    """Renaming a Product Folder keeping all files and properly renaming them.
    Channel metadata are edited to replace the old Filename field with the new one.

    Parameters
    ----------
    current_folder : Union[str, Path]
        current Product Folder, full path
    new_folder : Union[str, Path]
        desired Product Folder, full path
    """

    current_folder = Path(current_folder)
    new_folder = Path(new_folder)

    if not current_folder.exists():
        raise InvalidProductFolder(
            f"Current Product Folder does not exist {current_folder}"
        )

    if not current_folder.is_dir():
        raise InvalidProductFolder(
            f"Current Product Folder is not a directory {current_folder}"
        )

    if new_folder.exists():
        raise InvalidProductFolder(
            f"New Product Folder location already exist {new_folder}"
        )

    old_name = current_folder.name
    new_name = new_folder.name

    # rename full folder
    current_folder.rename(new_folder)
    assert new_folder.exists()

    # renaming all files containing the old product name
    for file in new_folder.iterdir():
        if old_name in str(file):
            if file.suffix == METADATA_EXTENSION:
                # editing the Filename field inside channel metadata XML files
                _update_metadata_filename_field(
                    file=file, old_name=old_name, new_name=new_name
                )
            old_filename = file.name
            file.rename(file.with_name(old_filename.replace(old_name, new_name)))


def _update_metadata_filename_field(file: Path, old_name: str, new_name: str) -> None:
    """Updating the channel metadata XML Filename node changing the old name with the new one.
    This operation is needed to generate a valid renamed product folder.

    Parameters
    ----------
    file : Path
        Path to the metadata XML file
    old_name : str
        old name string pattern to be replaced
    new_name : str
        new name to be inserted
    """
    xml_string = file.read_text()
    if xml_string.find(old_name) != -1:
        # if at least 1 occurrence of the old name is found, change them
        # else leave the file like it is
        xml_string = xml_string.replace(old_name, new_name)
    file.write_text(xml_string)
