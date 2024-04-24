# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Manifest module
---------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from lxml import etree

from arepytools.io.productfolder_layout import RasterExtensions

_VERSION = "2.1"


class InvalidManifestFilePath(RuntimeError):
    """If file path does not exist"""


@dataclass
class Manifest:
    """Product folder manifest document dataclass"""

    version: str = _VERSION
    description: Optional[str] = None
    datafile_extension: Optional[Union[str, RasterExtensions]] = None

    def __post_init__(self) -> None:
        if self.datafile_extension is None:
            self.datafile_extension = RasterExtensions.RAW
        self.datafile_extension = RasterExtensions(self.datafile_extension)

    def write(self, file_path: Union[str, Path]) -> None:
        """Writing manifest file to disk.

        Parameters
        ----------
        file_path : Union[str, Path]
            path of the file to be written, comprehensive of file name
        """

        file_path = Path(file_path)

        manifest_xml = etree.Element("AresysProductManifest")
        manifest_xml.set("Version", self.version)
        if self.description is not None:
            etree.SubElement(manifest_xml, "ProductDescription").text = self.description

        assert self.datafile_extension is not None and isinstance(
            self.datafile_extension, RasterExtensions
        )
        etree.SubElement(manifest_xml, "DataFileExtension").text = (
            self.datafile_extension.value
        )

        tree = etree.ElementTree(manifest_xml)
        tree.write(
            str(file_path),
            pretty_print=True,
            xml_declaration=True,
            encoding="utf-8",
        )

    @staticmethod
    def from_file(file_path: Union[str, Path]) -> Manifest:
        """Reading manifest xml document from file.

        Parameters
        ----------
        file_path : Union[str, Path]
            path to the xml manifest file

        Returns
        -------
        Manifest
            Manifest dataclass containing all the info from loaded file
        """

        file_path = Path(file_path)

        if not file_path.is_file():
            raise InvalidManifestFilePath("Input file does not exist")

        # casting Path to string when loading due to bug with PosixPath on Linux
        manifest_root = etree.parse(str(file_path)).getroot()
        version = manifest_root.values()[0]
        try:
            description = manifest_root.find("ProductDescription").text
        except AttributeError:
            description = None
        try:
            extension = manifest_root.find("DataFileExtension").text
        except AttributeError:
            extension = RasterExtensions.RAW

        manifest = Manifest(
            version=version, description=description, datafile_extension=extension
        )

        return manifest
