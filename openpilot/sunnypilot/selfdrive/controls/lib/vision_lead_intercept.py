"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Vision Lead Intercept (ICBM + stock longitudinal only)

Some stock ACC systems filter out stopped / very slow vehicles while the set speed is high,
and only start tracking them once the car is close and slow. This planner watches the primary
vision lead from modelV2.leadsV3 and, when it sees a slow or stopped lead far ahead, produces a
descending vTarget that ICBM uses to walk the stock set speed down to the minimum the stock
system accepts (20 mph / 30 km/h). The car then arrives at the lead already at the minimum set
speed, and stock longitudinal performs the final approach and stop.

States:
  idle         - nothing to do, outputs V_CRUISE_UNSET so the planner ignores us.
  intercepting - walking the target down along a comfortable braking curve.
  holding      - released: the stock set speed is at the minimum, or the lead is close enough that
                 stock should be tracking it. The target is pinned to the minimum set speed so ICBM
                 finishes (or keeps) the set speed at the minimum and does not raise it back while the
                 lead is still ahead. A lower set speed never fights stock braking, so stock
                 longitudinal handles the approach and stop.
  resuming     - the car came to a stop (carState.standstill, the same signal as the standstill timer)
                 while holding. The target stays at the minimum while stopped; once the car is moving
                 again the intercept fully releases. ICBM then walks the set speed back up to the
                 driver's last set speed, or to the speed limit when Speed Limit Assist is on, since
                 both already feed the planner's target.
                 The walk up waits until the car moves because ICBM raises the set speed with RES/+,
                 which on many stock systems (HKG included) also resumes from a standstill hold.

Before the car stops, the intercept also fully releases once the lead drives away or is lost.
In any state, disengaging or the driver pressing RES/+ fully releases.
"""
import math

import openpilot.cereal.messaging as messaging
from opendbc.car import structs
from opendbc.car.structs import car
from openpilot.common.constants import CV
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET
from openpilot.sunnypilot import PARAMS_UPDATE_PERIOD
from openpilot.sunnypilot.selfdrive.car.intelligent_cruise_button_management.helpers import get_minimum_set_speed

PARAM = "IcbmVisionLeadIntercept"

ButtonType = car.CarState.ButtonEvent.Type
RAISE_BUTTONS = (ButtonType.accelCruise, ButtonType.resumeCruise)  # driver presses that cancel the intercept

# Lead detection
LEAD_PROB_MIN = 0.7                  # minimum model lead probability to consider the lead real
LEAD_V_MAX = 12.5 * CV.MPH_TO_MS     # absolute lead speed below which we consider intercepting
LEAD_V_RELEASE = 20. * CV.MPH_TO_MS  # lead speed above which it is considered to be driving away (hysteresis)
LEAD_DEBOUNCE_TIME = 0.5             # s, lead must satisfy the conditions this long before we act
LEAD_LOST_TIME = 2.0                 # s, lead must be missing this long before we fully release

# Distances are model lead x, i.e. measured from the device, not the front bumper.
STOCK_TRACKING_DIST = 35.            # m, closer than this we assume stock is tracking the lead
ARRIVE_DIST = 45.                    # m, distance at which we want the set speed to be at the minimum

# Target profile
A_COMFORT = 0.8                      # m/s^2, braking curve used to plan the set speed descent
MAX_TARGET_RAMP = 1.5                # m/s^2, max rate the target itself may descend per second
SPEED_MARGIN = 2. * CV.MPH_TO_MS     # only intercept when ego and set speed are this far above the minimum
AT_MIN_MARGIN = 1. * CV.MPH_TO_MS    # stock set speed within this of the minimum counts as "at the minimum"
RESUME_MIN_SPEED = 2.                # m/s, after a stop, only release (and walk the set speed up) once moving faster than this


class VisionLeadState:
  idle = 0
  intercepting = 1
  holding = 2
  resuming = 3


class VisionLeadIntercept:
  def __init__(self, CP: structs.CarParams, CP_SP: structs.CarParamsSP):
    self.CP = CP
    self.CP_SP = CP_SP
    self.params = Params()
    self.frame = -1

    self.enabled = False
    self.is_metric = False
    self._read_params()

    self.state = VisionLeadState.idle
    self.lead_valid_timer = 0.
    self.lead_lost_timer = 0.
    self.cancelled = False  # driver pressed RES/+: ignore this lead until it is gone

    self.v_target = V_CRUISE_UNSET
    self.output_v_target = V_CRUISE_UNSET
    self.output_a_target = 0.

  @property
  def available(self) -> bool:
    # Only with stock longitudinal and ICBM actually managing the set speed (pcmCruiseSpeed is
    # cleared by ICBM init when the ICBM toggle is on and the platform supports it).
    return not self.CP.openpilotLongitudinalControl and not self.CP_SP.pcmCruiseSpeed

  @property
  def is_active(self) -> bool:
    return self.state != VisionLeadState.idle

  def _read_params(self) -> None:
    self.enabled = self.params.get_bool(PARAM)
    self.is_metric = self.params.get_bool("IsMetric")

  def _reset(self) -> None:
    self.state = VisionLeadState.idle
    self.v_target = V_CRUISE_UNSET

  def _start_intercept(self, v_ego: float, v_cruise: float, v_cruise_stock: float) -> None:
    self.state = VisionLeadState.intercepting
    # Start the descent from where the car is now, never above the current set speeds
    self.v_target = min(v_ego, v_cruise, v_cruise_stock)

  def update(self, sm: messaging.SubMaster, long_enabled: bool, long_override: bool, v_ego: float, v_cruise: float) -> None:
    self.frame += 1
    if self.frame % int(PARAMS_UPDATE_PERIOD / DT_MDL) == 0:
      self._read_params()

    min_set_speed = get_minimum_set_speed(self.is_metric) * (CV.KPH_TO_MS if self.is_metric else CV.MPH_TO_MS)
    # Actual stock set speed, which is what ICBM walks toward our target
    v_cruise_stock = sm['carState'].cruiseState.speedCluster

    lead = sm['modelV2'].leadsV3[0] if len(sm['modelV2'].leadsV3) else None
    lead_present = lead is not None and len(lead.x) > 0 and lead.prob >= LEAD_PROB_MIN
    lead_d = lead.x[0] if lead_present else 0.
    lead_v = lead.v[0] if lead_present else 0.

    # A slow lead that is still far enough away that stock is likely filtering it
    lead_target = lead_present and lead_v < LEAD_V_MAX and lead_d > STOCK_TRACKING_DIST
    self.lead_valid_timer = self.lead_valid_timer + DT_MDL if lead_target else 0.
    self.lead_lost_timer = 0. if lead_present else self.lead_lost_timer + DT_MDL

    lead_gone = self.lead_lost_timer > LEAD_LOST_TIME or (lead_present and lead_v > LEAD_V_RELEASE)
    if lead_gone:
      self.cancelled = False

    # Driver pressed RES/+ while we were active: respect it and stand down for this lead.
    # Button presses, not set speed changes, so Speed Limit Assist moving the set speed does not cancel us.
    if self.is_active and any(b.type in RAISE_BUTTONS and b.pressed for b in sm['carState'].buttonEvents):
      self.cancelled = True

    # A new slow lead far ahead worth intercepting (from idle, or while resuming after a stop)
    can_intercept = (self.lead_valid_timer >= LEAD_DEBOUNCE_TIME and not long_override and
                     v_ego > min_set_speed + SPEED_MARGIN and v_cruise_stock > min_set_speed + SPEED_MARGIN)

    if not (self.enabled and self.available and long_enabled) or self.cancelled:
      self._reset()

    elif lead_gone and self.state in (VisionLeadState.intercepting, VisionLeadState.holding):
      # Lead left before we stopped behind it: hand the set speed back to the driver
      self._reset()

    elif self.state == VisionLeadState.idle:
      if can_intercept:
        self._start_intercept(v_ego, v_cruise, v_cruise_stock)

    elif self.state == VisionLeadState.holding:
      self.v_target = min_set_speed
      if sm['carState'].standstill:
        # Stopped behind the lead (same standstill signal as the standstill timer)
        self.state = VisionLeadState.resuming

    elif self.state == VisionLeadState.resuming:
      if sm['carState'].standstill or v_ego < RESUME_MIN_SPEED:
        # Still stopped: hold the minimum so ICBM sends no RES/+ presses that could resume the car
        self.v_target = min_set_speed
      else:
        # Moving again: release, so ICBM walks back up to the driver's set speed (or the speed limit with SLA on)
        self._reset()

    elif self.state == VisionLeadState.intercepting:
      if v_cruise_stock <= min_set_speed + AT_MIN_MARGIN or (lead_present and lead_d <= STOCK_TRACKING_DIST):
        # Stock is at the minimum or close enough to track the lead: let stock take over
        self.state = VisionLeadState.holding
        self.v_target = min_set_speed
      elif lead_present:
        # Highest speed from which a comfortable A_COMFORT brake reaches the minimum set speed by ARRIVE_DIST.
        # The lead's own motion is ignored, which only makes this conservative.
        v_curve = math.sqrt(min_set_speed ** 2 + 2. * A_COMFORT * max(lead_d - ARRIVE_DIST, 0.))
        v_desired = max(min(v_curve, v_cruise), min_set_speed)
        # Descend smoothly and never step back up, so the set speed does not bounce
        self.v_target = min(self.v_target, max(v_desired, self.v_target - MAX_TARGET_RAMP * DT_MDL))

    self.output_v_target = self.v_target if self.is_active else V_CRUISE_UNSET
    self.output_a_target = -A_COMFORT if self.state == VisionLeadState.intercepting else 0.
