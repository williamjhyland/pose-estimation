# wifi-sensor/src/__init__.py
from viam.components.sensor import Sensor
from viam.resource.registry import Registry, ResourceCreatorRegistration
from .kasa_smart_plug import MySensor


# Registry.register_resource_creator(Sensor.SUBTYPE, MySensor.MODEL, ResourceCreatorRegistration(MySensor.new))