# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
IO Channel Iteration module
---------------------------
"""
from typing import Callable, Iterator, List, Tuple, Union

from arepytools.io.io_support import read_metadata
from arepytools.io.metadata import EPolarization, MetaData
from arepytools.io.productfolder2 import ProductFolder2

MetaDataFilter = Callable[[MetaData], bool]
EPolLike = Union[str, EPolarization]


class InvalidFilter(RuntimeError):
    """Invalid filter"""


class SwathIDFilter:
    """SwathID filtering class"""

    def __init__(
        self, polarization: Union[EPolLike, List[EPolLike]] = None, swath: str = None
    ):
        """Filtering metadata file searching for the selected polarization and/or swath.

        Parameters
        ----------
        polarization : Union[EPolLike, List[EPolLike]], optional
            polarizations to be filtered, it can be a single value or a list of polarizations, by default None
        swath : str, optional
            swath name to be filtered, by default None

        Raises
        ------
        InvalidFilter
            both filtering fields are not defined
        """
        if polarization is None and swath is None:
            raise InvalidFilter("No filtering fields have been defined")

        if not isinstance(polarization, list) and polarization is not None:
            polarization = [polarization]
        self.polarization = (
            [EPolarization(p) for p in polarization]
            if polarization is not None
            else None
        )

        self.swath = swath

    def __call__(self, metadata: MetaData) -> bool:
        """Filtering out the input metadata file (first channel only) to match the polarization and/or swath name
        as defined during class initialization.

        Parameters
        ----------
        metadata : MetaData
            metadata object to be checked

        Returns
        -------
        bool
            boolean matching result
        """
        swath_info = metadata.get_swath_info()
        if self.polarization is not None:
            pol_condition = swath_info.polarization in self.polarization
            if self.swath is not None:
                sw_condition = swath_info.swath == self.swath
                return pol_condition and sw_condition

            return pol_condition

        if self.swath is not None:
            return swath_info.swath == self.swath


def iter_channels_generator(
    product: ProductFolder2,
    filter_func: MetaDataFilter = None,
) -> Iterator[Tuple[int, MetaData]]:
    """Channel iteration generator that yields channel id and channel metadata.
    If a filter MetaDataFilter-like function is provided, the output yielded is restricted only
    to channels matching the filtering conditions.

    Parameters
    ----------
    product : ProductFolder2
        product folder from which to get the channels
    filter_func : MetaDataFilter, optional
        filtering MetaDataFilter-like function, by default None

    Returns
    -------
    Tuple[int, MetaData]
        channel id,
        channel metadata object

    Yields
    ------
    Iterator[Tuple[int, MetaData]]
        channel id,
        channel metadata object
    """
    for ch_index in product.get_channels_list():
        metadata = read_metadata(product.get_channel_metadata(ch_index))

        if filter_func is None or filter_func(metadata):
            yield ch_index, metadata


def iter_channels(
    product: ProductFolder2,
    polarization: Union[EPolLike, List[EPolLike]] = None,
    swath: str = None,
) -> Iterator[Tuple[int, MetaData]]:
    """Channels iterator with optional SwathID filter pre-configured.

    Parameters
    ----------
    product : ProductFolder2
        product folder from which to get the channels
    polarization : Union[EPolLike, List[EPolLike]], optional
        polarizations to be filtered, it can be a single value or a list of polarizations, by default None
    swath : str, optional
        swath name to be filtered, by default None

    Returns
    -------
    Tuple[int, MetaData]
        channel id,
        channel metadata object

    Yields
    ------
    Iterator[Tuple[int, MetaData]]
        channel id,
        channel metadata object
    """
    filtering = None
    if polarization is not None or swath is not None:
        filtering = SwathIDFilter(polarization=polarization, swath=swath)

    yield from iter_channels_generator(product=product, filter_func=filtering)
