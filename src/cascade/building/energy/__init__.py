"""Common-area energy smart switching for towers (ask 3): code-bounded policy, ML models, replay and LA calendar.

Modules
  rules       code floors per common-area zone type (common_area_rules.json, cited; not a compliance tool)
  policy      deterministic per-zone lighting state machine + pending-approval Proposal log (ML may only raise light)
  occupancy   M1 presence classifier and M2 1 h presence forecaster on REAL ROBOD, UCI 357 leakage check, UCI 864 PIR
  forecast    M3 day-ahead garage load forecaster on REAL BDG2 parking meters vs lag-24 / lag-168 naive
  tariff      LADWP A-2 Rate B TOU periods and charges, SCE TOU windows, DR / Flex Alert windows
  simulate    REAL ROBOD replay (recorded kWh x level) and the SEMI-SYNTHETIC tower year
  inventory   SEMI-SYNTHETIC common-area inventory for the synthetic 32-floor tower ([ASSUMPTION] W, ft2, fc)
  calendar_la 12-month LA mitigation calendar (NOAA normals, NOAA GML daylight, tariff/DR seasons, holidays, DST)
  data        downloads, manifest and loaders

Safety contract: the AI proposes and a human approves. Nothing here writes to emergency, exit-sign or egress-floor
circuits; a sensor fault sends lighting to full ON.
"""

from .data import LA_TZ

__all__ = ["LA_TZ"]
