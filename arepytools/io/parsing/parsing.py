# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Parsing
-------
"""

from typing import Any, Type

import xsdata.exceptions
from xsdata.formats.dataclass.context import XmlContext
from xsdata.formats.dataclass.parsers import XmlParser
from xsdata.formats.dataclass.serializers import XmlSerializer
from xsdata.formats.dataclass.serializers.config import SerializerConfig

_CONTEXT = XmlContext()
_PARSER = XmlParser(context=_CONTEXT)
_SERIALIZER_CONFIGURATION = SerializerConfig(pretty_print=True, encoding="utf-8")
_SERIALIZER = XmlSerializer(context=_CONTEXT, config=_SERIALIZER_CONFIGURATION)


class ParsingError(RuntimeError):
    """Raise when the XML parsing fails"""


def parse(xml_string: str, model: Type) -> Any:
    """Parse a string according to an XSD model

    Parameters
    ----------
    xml_string : str
        input xml string to parse
    model : Type
        xsd model type

    Returns
    -------
    Any
        The content as a structure of type model

    Raises
    ------
    ParsingError
        in case the xml_string is incompatible with the XSD model
    """
    try:
        model_obj = _PARSER.from_string(xml_string, model)
    except xsdata.exceptions.ParserError as exc:
        raise ParsingError from exc

    return model_obj


def serialize(model: Any, **kwargs) -> str:
    """Serialize an XSD object to a string

    Parameters
    ----------
    model :
        Object to serialize

    kwargs are forwarded to serializer render method

    Returns
    -------
    str
        XML string
    """
    return _SERIALIZER.render(model, **kwargs)
