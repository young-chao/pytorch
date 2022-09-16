# This file was generated by jschema_to_python version 1.2.3.

import attr


@attr.s
class Tool(object):
    """The analysis tool that was run."""

    driver = attr.ib(metadata={"schema_property_name": "driver"})
    extensions = attr.ib(default=None, metadata={"schema_property_name": "extensions"})
    properties = attr.ib(default=None, metadata={"schema_property_name": "properties"})


# flake8: noqa
