# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Aresys Point Target File XSD Models
-----------------------------------
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class PointTargetsTargetType(Enum):
    VALUE_0 = 0
    VALUE_1 = 1
    VALUE_2 = 2


@dataclass
class ValType:
    class Meta:
        name = "valType"

    value: Optional[float] = field(
        default=None,
        metadata={
            "required": True,
        },
    )
    n: Optional[int] = field(
        default=None,
        metadata={
            "name": "N",
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class ValTypeComplex:
    class Meta:
        name = "valTypeComplex"

    re: Optional[float] = field(
        default=None,
        metadata={
            "type": "Element",
            "required": True,
        },
    )
    im: Optional[float] = field(
        default=None,
        metadata={
            "type": "Element",
            "required": True,
        },
    )
    n: Optional[int] = field(
        default=None,
        metadata={
            "name": "N",
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class Rcstype:
    class Meta:
        name = "RCSType"

    val: List[ValTypeComplex] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "min_occurs": 2,
            "max_occurs": 2,
        },
    )


@dataclass
class TargetTagType:
    coord: Optional["TargetTagType.Coord"] = field(
        default=None,
        metadata={
            "name": "Coord",
            "type": "Element",
            "required": True,
        },
    )
    rcs_h: Optional[Rcstype] = field(
        default=None,
        metadata={
            "name": "RCS_H",
            "type": "Element",
            "required": True,
        },
    )
    rcs_v: Optional[Rcstype] = field(
        default=None,
        metadata={
            "name": "RCS_V",
            "type": "Element",
            "required": True,
        },
    )
    delay: Optional["TargetTagType.Delay"] = field(
        default=None,
        metadata={
            "name": "Delay",
            "type": "Element",
            "required": True,
        },
    )
    number: Optional[int] = field(
        default=None,
        metadata={
            "name": "Number",
            "type": "Attribute",
            "required": True,
        },
    )

    @dataclass
    class Coord:
        val: List[ValType] = field(
            default_factory=list,
            metadata={
                "type": "Element",
                "min_occurs": 3,
                "max_occurs": 3,
            },
        )

    @dataclass
    class Delay:
        val: Optional[ValType] = field(
            default=None,
            metadata={
                "type": "Element",
                "required": True,
            },
        )


@dataclass
class PointTargets:
    """
    :ivar target_type: 0: LLH, 1: ECEF, 2: Normalized
    :ivar target:
    :ivar ntarget:
    """

    target_type: Optional[PointTargetsTargetType] = field(
        default=None,
        metadata={
            "name": "TargetType",
            "type": "Element",
            "required": True,
        },
    )
    target: List[TargetTagType] = field(
        default_factory=list,
        metadata={
            "name": "Target",
            "type": "Element",
            "min_occurs": 1,
        },
    )
    ntarget: Optional[int] = field(
        default=None,
        metadata={
            "name": "Ntarget",
            "type": "Element",
            "required": True,
        },
    )
