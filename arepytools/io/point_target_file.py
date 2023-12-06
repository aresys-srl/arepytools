# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Point Target File Module
------------------------
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np

import arepytools.io.io_support as support
from arepytools.geometry.conversions import llh2xyz
from arepytools.io.parsing.parsing import parse, serialize
from arepytools.io.point_target_models import aresys_point_target_models as models

IDLike = Union[int, str]
IDLikeList = Union[IDLike, List[IDLike]]


def read_point_targets_file(
    xml_file: Union[str, Path]
) -> Dict[str, support.NominalPointTarget]:
    """Reading a point target xml file to retrieve information about point target location, radiation cross
    sections and delay.

    Parameters
    ----------
    xml_file : Union[str, Path]
        path to the xml point targets file

    Returns
    -------
    Dict[str, support.NominalPointTarget]
        Dict of NominalPointTarget dataclasses, one for each target id detected (id are the keys)
    """

    parsed_xml = _parse_xml_with_models(xml_path=xml_file)

    return _translate_point_target_file_from_model(data_model=parsed_xml)


def write_point_targets_file(
    filename: Union[str, Path],
    point_targets: Union[List[support.NominalPointTarget], support.NominalPointTarget],
    target_type: Union[int, support.CoordinatesType],
    point_targets_ids: Union[IDLike, IDLikeList] = None,
) -> None:
    """Writing PointTargetFile XML to disk based on input point targets data.

    Parameters
    ----------
    filename : Union[str, Path]
        path to the xml file to be written, xml suffix must be included
    point_targets : Union[List[support.NominalPointTarget], support.NominalPointTarget]
        list of NominalPointTarget point targets to be written or single NominalPointTarget object
    target_type : Union[int, support.CoordinatesType]
        int or CoordinatesType value to specify the point target coordinates format of the provided data
    point_targets_ids : Union[IDLike, IDLikeList], optional
        list of point target IDs as integers or integers-like strings, if None IDs are created starting from 1 to
        the actual number of point targets elements provided, by default None
    """
    target_type = support.CoordinatesType(target_type)
    filename = Path(filename)

    if filename.exists():
        raise support.InvalidPointTargetError(f"Path already exists {filename}")

    if filename.suffix != ".xml":
        raise support.InvalidPointTargetError(f"File extension is not XML {filename}")

    if isinstance(point_targets, support.NominalPointTarget):
        point_targets = [point_targets]

    # generating point targets ids if not provided (starting from 1)
    if point_targets_ids is None:
        point_targets_ids = list(range(1, len(point_targets) + 1))
    else:
        if np.size(point_targets_ids) == 1:
            point_targets_ids = [point_targets_ids]

    if np.size(point_targets_ids) != np.size(point_targets):
        raise support.PointTargetDimensionsMismatchError(
            f"point targets number {np.size(point_targets)} != point target ids number {np.size(point_targets_ids)}"
        )

    # composing single point target nodes for each node
    point_target_nodes = [
        _convert_custom_dataclass_to_xml_model(data=p[0], data_id=int(p[1]))
        for p in zip(point_targets, point_targets_ids)
    ]

    # composing main XML node
    main_node = models.PointTargets(
        target_type=models.PointTargetsTargetType(target_type.value),
        target=point_target_nodes,
        ntarget=len(point_target_nodes),
    )

    # writing XML file
    xml_string = serialize(main_node)
    filename.write_text(xml_string, encoding="utf-8")


def _parse_xml_with_models(xml_path: Union[str, Path]) -> models.PointTargets:
    """Parsing Point Target File xml using the xsdata model generated from original xsd file.

    Parameters
    ----------
    xml_path : Union[str, Path]
        path to the xml file

    Returns
    -------
    models.PointTargets
        point target xml document as a PointTargets dataclass
    """
    xml = Path(xml_path).read_text(encoding="utf-8")
    doc = parse(xml_string=xml, model=models.PointTargets)

    return doc


def _translate_point_target_file_from_model(
    data_model: models.PointTargets,
) -> Dict[str, support.NominalPointTarget]:
    """Translate the main xsdata model (PointTargets node) parsed from XML file to a dict of
    custom NominalPointTarget dataclasses, one for each point target.

    Parameters
    ----------
    data_model : models.PointTargets
        PointTargets dataclass model node from parsing xml file

    Returns
    -------
    Dict[str, support.NominalPointTarget]
        dict of NominalPointTarget objects as values, key are the corresponding target IDs
    """

    assert isinstance(data_model, models.PointTargets)
    coord_type = support.CoordinatesType(data_model.target_type.value)
    targets = data_model.target

    targets = dict(
        [
            _convert_xml_model_to_custom_dataclass(target, coord_type=coord_type)
            for target in targets
        ]
    )

    return targets


def _convert_xml_model_to_custom_dataclass(
    target: models.TargetTagType, coord_type: support.CoordinatesType
) -> Tuple[str, support.NominalPointTarget]:
    """Conversion function from TargetType xsdata model dataclass to NominalPointTarget custom dataclass.
    If coordinates are expressed in LLH format, they are converted to ECEF.
    Returning also the point target ID.

    Parameters
    ----------
    target : models.TargetTagType
        xsdata model TargetTagType from xml parsing
    coord_type : support.CoordinatesType
        coordinate type of each target point

    Returns
    -------
    Tuple[str, support.NominalPointTarget]
        target id,
        target dataclass
    """

    # checking the existence of requested fields
    assert target.coord is not None
    assert target.rcs_h is not None
    assert target.rcs_v is not None
    assert target.delay is not None

    # retrieving information
    coord = _coords_from_model(target.coord)
    if coord_type == support.CoordinatesType.LLH:
        coord = llh2xyz(coord).squeeze().tolist()
    elif coord_type == support.CoordinatesType.NORMALIZED:
        raise NotImplementedError("Point Target normalized coordinates not supported")
    rcs_hh, rcs_hv = _rcs_from_model(target.rcs_h)
    rcs_vv, rcs_vh = _rcs_from_model(target.rcs_v)
    delay = _delay_from_model(target.delay)

    # composing custom dataclass
    point_target = support.NominalPointTarget(
        xyz_coordinates=coord,
        rcs_hh=rcs_hh,
        rcs_hv=rcs_hv,
        rcs_vv=rcs_vv,
        rcs_vh=rcs_vh,
        delay=delay,
    )

    return (str(target.number), point_target)


def _convert_custom_dataclass_to_xml_model(
    data: support.NominalPointTarget, data_id: int
) -> models.TargetTagType:
    """Converting custom NominalPointTarget dataclass to xsdata TargetType model dataclass for writing purposes.

    Parameters
    ----------
    data : support.NominalPointTarget
        point target data in custom NominalPointTarget form
    data_id : int
        point target id number

    Returns
    -------
    models.TargetType
        TargetType xsdata model dataclass corresponding to input information
    """

    coord_node = models.TargetTagType.Coord(
        [
            models.ValType(value=item, n=n + 1)
            for n, item in enumerate(data.xyz_coordinates)
        ]
    )
    rcs_h = data.rcs_hh, data.rcs_hv
    rcs_h_node = models.Rcstype(
        [
            models.ValTypeComplex(re=item.real, im=item.imag, n=n + 1)
            for n, item in enumerate(rcs_h)
        ]
    )
    rcs_v = data.rcs_vv, data.rcs_vh
    rcs_v_node = models.Rcstype(
        [
            models.ValTypeComplex(re=item.real, im=item.imag, n=n + 1)
            for n, item in enumerate(rcs_v)
        ]
    )
    delay_node = models.TargetTagType.Delay(models.ValType(value=data.delay, n=1))

    return models.TargetTagType(
        coord=coord_node,
        rcs_h=rcs_h_node,
        rcs_v=rcs_v_node,
        delay=delay_node,
        number=data_id,
    )


def _rcs_from_model(data_model: models.Rcstype) -> Tuple[complex, complex]:
    """Converting input RCS model dataclass to rcs values.

    Parameters
    ----------
    data_model : models.Rcstype
        target rcs data model

    Returns
    -------
    Tuple[complex, complex]
        rcs same polarization,
        rcs cross polarization
    """
    assert isinstance(data_model, models.Rcstype)

    rcs = [(r.re + 1j * r.im, r.n) for r in data_model.val]
    # sorting values based on last tuple element that is the coordinate id in the xml file
    # i.e. 1=x, 2=y, 3=z
    rcs.sort(key=lambda x: x[-1])

    return [r[0] for r in rcs]


def _coords_from_model(data_model: models.TargetTagType.Coord) -> list:
    """Converting input model dataclass to list of point target coordinates.

    Parameters
    ----------
    data_model : models.TargetTagType.Coord
        target type coord data model

    Returns
    -------
    list
        list of coordinates [x, y, z]
    """
    assert isinstance(data_model, models.TargetTagType.Coord)

    coords = [(d.value, d.n) for d in data_model.val]
    # sorting values based on last tuple element that is the coordinate id in the xml file
    # i.e. 1=x, 2=y, 3=z
    coords.sort(key=lambda x: x[-1])

    return [c[0] for c in coords]


def _delay_from_model(data_model: models.TargetTagType.Delay) -> float:
    """Converting input model dataclass to signal delay.

    Parameters
    ----------
    data_model : models.TargetTagType.Delay
        target type delay data model

    Returns
    -------
    float
        point target delay
    """
    assert isinstance(data_model, models.TargetTagType.Delay)
    return data_model.val.value
