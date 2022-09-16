# This file was generated by jschema_to_python version 1.2.3.

import attr


@attr.s
class RunAutomationDetails(object):
    """Information that describes a run's identity and role within an engineering system process."""

    correlation_guid = attr.ib(
        default=None, metadata={"schema_property_name": "correlationGuid"}
    )
    description = attr.ib(
        default=None, metadata={"schema_property_name": "description"}
    )
    guid = attr.ib(default=None, metadata={"schema_property_name": "guid"})
    id = attr.ib(default=None, metadata={"schema_property_name": "id"})
    properties = attr.ib(default=None, metadata={"schema_property_name": "properties"})


# flake8: noqa
