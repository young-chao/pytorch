# This file was generated by jschema_to_python version 1.2.3.

import attr


@attr.s
class ReportingDescriptor(object):
    """Metadata that describes a specific report produced by the tool, as part of the analysis it provides or its runtime reporting."""

    id = attr.ib(metadata={"schema_property_name": "id"})
    default_configuration = attr.ib(
        default=None, metadata={"schema_property_name": "defaultConfiguration"}
    )
    deprecated_guids = attr.ib(
        default=None, metadata={"schema_property_name": "deprecatedGuids"}
    )
    deprecated_ids = attr.ib(
        default=None, metadata={"schema_property_name": "deprecatedIds"}
    )
    deprecated_names = attr.ib(
        default=None, metadata={"schema_property_name": "deprecatedNames"}
    )
    full_description = attr.ib(
        default=None, metadata={"schema_property_name": "fullDescription"}
    )
    guid = attr.ib(default=None, metadata={"schema_property_name": "guid"})
    help = attr.ib(default=None, metadata={"schema_property_name": "help"})
    help_uri = attr.ib(default=None, metadata={"schema_property_name": "helpUri"})
    message_strings = attr.ib(
        default=None, metadata={"schema_property_name": "messageStrings"}
    )
    name = attr.ib(default=None, metadata={"schema_property_name": "name"})
    properties = attr.ib(default=None, metadata={"schema_property_name": "properties"})
    relationships = attr.ib(
        default=None, metadata={"schema_property_name": "relationships"}
    )
    short_description = attr.ib(
        default=None, metadata={"schema_property_name": "shortDescription"}
    )


# flake8: noqa
