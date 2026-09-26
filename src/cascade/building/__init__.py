"""Smart-building and skyscraper problem management on top of the grading engine (building_spec, D-015).

A building has floors, zones on floor plans, elements (drains, risers, panels, circuits) and sensors. Two graphs
link them: the water path (drains_to, above, adjacent) and the electrical tree (feeds). Every input becomes an
Observation; the problem manager groups observations along the graphs and in time, ranks root causes with named
additive factors, estimates time to escalation and keeps a status lifecycle.

Modules: model (data model), rules (threshold registry), graph, blueprint (store, plans, tickets), synthetic
(labelled SYNTHETIC demo tower), water, electrical, grading, problems.
"""

from .model import (AZIMUTH, Building, BuildingData, Edge, Element, ElectricalResult, Escalation, Factor, Floor, Observation,
                    Problem, ResponseFit, RootCause, Sensor, StatusChange, StormEvent, WaterResult, Zone, ZoneForecast)

__all__ = [
    "AZIMUTH", "Building", "BuildingData", "Edge", "Element", "ElectricalResult", "Escalation", "Factor", "Floor", "Observation",
    "Problem", "ResponseFit", "RootCause", "Sensor", "StatusChange", "StormEvent", "WaterResult", "Zone", "ZoneForecast",
]
