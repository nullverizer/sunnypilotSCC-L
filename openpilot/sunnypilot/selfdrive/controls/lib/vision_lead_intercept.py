"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Vision Lead Intercept (ICBM + stock longitudinal only, Hyundai/Kia/Genesis HDA2)

The problem: some stock ACC systems ignore stopped or very slow vehicles while the set speed is high, and
only start tracking them once the car is close and slow.

What this does, in order:
  1. Watches the primary vision lead (modelV2.leadsV3). When it is a confident, slow or stopped car that is
     still far ahead, it starts lowering a target speed along a gentle braking curve.
  2. ICBM sees that lower target and presses the cruise buttons to walk the stock set speed down to the
     minimum the stock system accepts (20 mph / 30 km/h), so the car arrives at the lead already slow.
  3. Once the set speed is at the minimum, or the lead is close enough for stock to be tracking it, we stop
     lowering and just keep the target at the minimum. Stock longitudinal then does the final approach and stop.
  4. When the car comes to a stop (carState.standstill, the same signal as the standstill timer), we let go
     completely. The stock system holds the stop, and ICBM walks the set speed back up to the driver's last
     set speed, or to the speed limit when Speed Limit Assist is on (both already feed the planner's target).

We also let go completely, without waiting for a stop, when: the lead pulls away (speeds up, or is above
15 mph) or is lost, cruise is not
engaged, the driver presses the gas, or the driver presses RES/+.

States:
  idle         - nothing to do; outputs V_CRUISE_UNSET so the planner ignores us.
  intercepting - lowering the target along the braking curve.
  holding      - target pinned at the minimum set speed until the car stops or we let go.
"""
import math

import openpilot.cereal.messaging as messaging
from opendbc.car import structs
from opendbc.car.structs import car
from opendbc.car.hyundai.values import HyundaiFlags
from openpilot.common.constants import CV
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET
from openpilot.sunnypilot import PARAMS_UPDATE_PERIOD
from openpilot.sunnypilot.selfdrive.car.intelligent_cruise_button_management.helpers import get_minimum_set_speed

# ======================================================================================
# SECTION 1 - Settings and tuning numbers
# All the numbers that decide how the feature behaves live here, so they are easy to find and tune.
# ======================================================================================
PARAM = "IcbmVisionLeadIntercept"  # the on/off toggle (off by default)

ButtonType = car.CarState.ButtonEvent.Type
RAISE_BUTTONS = (ButtonType.accelCruise, ButtonType.resumeCruise)  # driver presses that cancel the intercept

# What counts as a lead worth intercepting: a car we are confident about, that is slow or stopped.
LEAD_PROB_MIN = 0.7                  # the model must be at least this sure the lead is real
LEAD_V_MAX = 12.5 * CV.MPH_TO_MS     # lead's own ground speed must be below this to start (stopped or crawling)
LEAD_V_RELEASE = 15. * CV.MPH_TO_MS  # lead faster than this means it is driving away, so we give up (higher than
                                     # LEAD_V_MAX on purpose so a lead hovering near the limit does not flicker)
LEAD_A_RELEASE = 0.7                 # m/s^2, lead speeding up harder than this also means it is driving away, even while
                                     # still slow, so we do not keep slowing down for something that is leaving
LEAD_DEPART_TIME = 0.5               # s, the lead must keep pulling away this long before we give up (ignores noise)
LEAD_DEBOUNCE_TIME = 0.5             # s, the lead must keep qualifying this long before we act (ignores blips)
LEAD_LOST_TIME = 2.0                 # s, the lead must be missing this long before we treat it as gone
LEAD_LOST_TIME_CLOSE = 8.0           # s, same, but when the lead was last seen close: the model often drops a lead that is
                                     # right in front of us, so wait much longer before assuming it left

# Distances are the model's lead distance, measured from the device rather than the front bumper.
STOCK_TRACKING_DIST = 35.            # m, closer than this we assume the stock system can see the lead and takes over
ARRIVE_DIST = 45.                    # m, we want the set speed down at the minimum by the time the lead is this far

# How the set speed comes down.
A_COMFORT = 0.8                      # m/s^2, gentle braking curve used to decide how fast we need to be at each distance
MAX_TARGET_RAMP = 1.5                # m/s per second, the target never drops faster than this (keeps button presses calm)
SPEED_MARGIN = 2. * CV.MPH_TO_MS     # only start if both the car and the set speed are at least this far above the minimum
AT_MIN_MARGIN = 1. * CV.MPH_TO_MS    # a set speed within this of the minimum counts as "already at the minimum"


# ======================================================================================
# SECTION 2 - Which cars may use this
# Only Hyundai / Kia / Genesis cars with HDA2 (CAN-FD, the camera sends the steering command). That is the
# only platform this was designed and tuned for, so every other car keeps the feature off.
# ======================================================================================
def is_hkg_hda2(CP: structs.CarParams) -> bool:
  return CP.brand == "hyundai" and bool(CP.flags & HyundaiFlags.CANFD_LKA_STEER_MSG)


# ======================================================================================
# SECTION 3 - The three states
#   idle         - not doing anything; hands the planner a huge "no opinion" speed.
#   intercepting - slowly lowering the target so the set speed comes down before we reach the lead.
#   holding      - the set speed is at (or heading to) the minimum, kept there until the car stops or the lead leaves.
# ======================================================================================
class VisionLeadState:
  idle = 0
  intercepting = 1
  holding = 2


class VisionLeadIntercept:
  # ------------------------------------------------------------------------------------
  # SECTION 4 - Setup: remember the car, start idle, and start with no lead seen yet.
  # ------------------------------------------------------------------------------------
  def __init__(self, CP: structs.CarParams, CP_SP: structs.CarParamsSP):
    self.CP = CP
    self.CP_SP = CP_SP
    self.params = Params()
    self.frame = -1

    self.enabled = False
    self.is_metric = False
    self._read_params()

    self.state = VisionLeadState.idle
    self.lead_valid_timer = 0.   # how long the lead has qualified in a row
    self.lead_lost_timer = 0.    # how long there has been no usable lead in a row
    self.lead_depart_timer = 0.  # how long the lead has been clearly pulling away in a row
    self.lead_d_last = math.inf  # distance at which the lead was last seen (inf = no lead remembered)
    self.cancelled = False       # driver pressed RES/+: leave this lead alone until it is gone

    self.v_target = V_CRUISE_UNSET         # our working target speed (m/s)
    self.output_v_target = V_CRUISE_UNSET  # what we hand to the planner (huge when idle = no opinion)
    self.output_a_target = 0.

  # ------------------------------------------------------------------------------------
  # SECTION 5 - Is the feature allowed to run on this car right now?
  # Needs an HDA2 car, the car's own cruise (not openpilot longitudinal), and ICBM turned on and managing the
  # set speed. ICBM's setup clears pcmCruiseSpeed exactly when it is on and supported.
  # ------------------------------------------------------------------------------------
  @property
  def available(self) -> bool:
    return is_hkg_hda2(self.CP) and not self.CP.openpilotLongitudinalControl and not self.CP_SP.pcmCruiseSpeed

  @property
  def is_active(self) -> bool:
    return self.state != VisionLeadState.idle

  def _read_params(self) -> None:
    # The toggle, and whether the dash shows mph or km/h (this decides the minimum set speed, 20 mph or 30 km/h).
    self.enabled = self.params.get_bool(PARAM)
    self.is_metric = self.params.get_bool("IsMetric")

  def _reset(self) -> None:
    # Give up completely: go idle and stop influencing the set speed, so ICBM goes back to the driver's set speed
    # (or the speed limit when Speed Limit Assist is on).
    self.state = VisionLeadState.idle
    self.v_target = V_CRUISE_UNSET

  def _start_intercept(self, v_ego: float, v_cruise: float, v_cruise_stock: float) -> None:
    # Begin lowering the target from where we are now, never from above the current speed or either set speed.
    self.state = VisionLeadState.intercepting
    self.v_target = min(v_ego, v_cruise, v_cruise_stock)

  # ------------------------------------------------------------------------------------
  # SECTION 6 - The main loop, called every planner cycle
  # ------------------------------------------------------------------------------------
  def update(self, sm: messaging.SubMaster, long_enabled: bool, long_override: bool, v_ego: float, v_cruise: float) -> None:
    # Step 1 - Housekeeping: re-read the settings every few seconds so a toggle change takes effect without a restart.
    self.frame += 1
    if self.frame % int(PARAMS_UPDATE_PERIOD / DT_MDL) == 0:
      self._read_params()

    # Step 2 - Gather the facts we need.
    # The lowest set speed the stock system accepts, in m/s.
    min_set_speed = get_minimum_set_speed(self.is_metric) * (CV.KPH_TO_MS if self.is_metric else CV.MPH_TO_MS)
    # The set speed currently shown on the dash. This is what ICBM actually moves with its button presses.
    v_cruise_stock = sm['carState'].cruiseState.speedCluster

    # The primary vision lead: how sure the model is, how far away it is, and how fast it is really moving.
    lead = sm['modelV2'].leadsV3[0] if len(sm['modelV2'].leadsV3) else None
    lead_present = lead is not None and len(lead.x) > 0 and lead.prob >= LEAD_PROB_MIN
    lead_d = lead.x[0] if lead_present else 0.
    lead_v = lead.v[0] if lead_present else 0.
    lead_a = lead.a[0] if lead_present and len(lead.a) > 0 else 0.

    # Step 3 - Is this lead worth intercepting? It must be slow or stopped, not pulling away, and still far enough
    # away that the stock system is probably ignoring it. It has to stay that way for a moment before we believe it.
    # "Pulling away" means it is already going faster than LEAD_V_RELEASE or it is speeding up hard.
    lead_pulling_away = lead_present and (lead_v > LEAD_V_RELEASE or lead_a > LEAD_A_RELEASE)
    lead_target = lead_present and lead_v < LEAD_V_MAX and lead_d > STOCK_TRACKING_DIST and not lead_pulling_away
    self.lead_valid_timer = self.lead_valid_timer + DT_MDL if lead_target else 0.
    self.lead_lost_timer = 0. if lead_present else self.lead_lost_timer + DT_MDL
    self.lead_depart_timer = self.lead_depart_timer + DT_MDL if lead_pulling_away else 0.

    # Step 4 - Has the lead gone? Yes if it has been clearly pulling away for a moment (so we stop slowing down for
    # something that is leaving), or we have not seen it for a while.
    # "A while" is a couple of seconds, but much longer if we last saw it close, because a lead right in front of us
    # often drops out of the model's view and we must not hand the set speed back while still closing on it.
    # Once it is gone, forget the cancel and the remembered distance so the next lead gets a fresh start.
    if lead_present:
      self.lead_d_last = lead_d
    lost_limit = LEAD_LOST_TIME_CLOSE if self.lead_d_last <= STOCK_TRACKING_DIST else LEAD_LOST_TIME
    lead_gone = self.lead_lost_timer > lost_limit or self.lead_depart_timer >= LEAD_DEPART_TIME
    if lead_gone:
      self.cancelled = False
      self.lead_d_last = math.inf

    # Step 5 - Driver cancel: if the driver presses RES/+ while we are active, they want the speed back, so we stand
    # down for this lead. We watch the button presses rather than the set speed, because Speed Limit Assist can
    # change the set speed on its own and that must not cancel us.
    if self.is_active and any(b.type in RAISE_BUTTONS and b.pressed for b in sm['carState'].buttonEvents):
      self.cancelled = True

    # Step 6 - Could we start a new intercept right now? We need a qualifying lead, and the car and the set speed
    # both clearly above the minimum (otherwise there is nothing to gain).
    can_intercept = (self.lead_valid_timer >= LEAD_DEBOUNCE_TIME and
                     v_ego > min_set_speed + SPEED_MARGIN and v_cruise_stock > min_set_speed + SPEED_MARGIN)

    # Step 7 - Decide what to do, checking the reasons to stand down first.
    if not (self.enabled and self.available and long_enabled) or long_override or self.cancelled:
      # Feature off, not allowed on this car, cruise not engaged, driver on the gas, or driver cancelled:
      # do nothing. (After a gas override ends we simply start fresh if the lead is still there.)
      self._reset()

    elif self.is_active and sm['carState'].standstill:
      # The car has come to a stop (same signal the standstill timer uses) and the stock system is holding it.
      # Release, so ICBM walks the set speed back up to the driver's last set speed, or the speed limit if
      # Speed Limit Assist is on. Raising the set speed while stopped is fine because the car itself holds the stop.
      self._reset()

    elif lead_gone and self.is_active:
      # The lead left (drove away, turned off, or was lost for good) before we stopped behind it: give the set speed
      # back to the driver.
      self._reset()

    elif self.state == VisionLeadState.idle:
      # Waiting for a slow lead far ahead.
      if can_intercept:
        self._start_intercept(v_ego, v_cruise, v_cruise_stock)

    elif self.state == VisionLeadState.holding:
      # Keep the set speed pinned at the minimum so ICBM does not raise it while the lead is still ahead.
      # A low set speed never fights the stock system's own braking.
      self.v_target = min_set_speed

    elif self.state == VisionLeadState.intercepting:
      if v_cruise_stock <= min_set_speed + AT_MIN_MARGIN or (lead_present and lead_d <= STOCK_TRACKING_DIST):
        # Either the dash set speed is already at the minimum, or the lead is close enough that the stock system
        # should be tracking it. Stop lowering and let the stock system handle the approach and the stop.
        self.state = VisionLeadState.holding
        self.v_target = min_set_speed
      elif lead_present:
        # Work out the fastest speed we could be doing at this distance and still brake gently down to the minimum
        # set speed by ARRIVE_DIST. (We ignore the lead's own movement, which only makes this more cautious.)
        v_curve = math.sqrt(min_set_speed ** 2 + 2. * A_COMFORT * max(lead_d - ARRIVE_DIST, 0.))
        v_desired = max(min(v_curve, v_cruise), min_set_speed)
        # That speed is never above the driver's own set speed and never below the minimum. Move the target toward
        # it, but only downward, and no faster than MAX_TARGET_RAMP, so the set speed steps down smoothly and
        # never bounces back up. (If the lead briefly drops out, the target just stays where it was.)
        self.v_target = min(self.v_target, max(v_desired, self.v_target - MAX_TARGET_RAMP * DT_MDL))

    # Step 8 - Publish. Idle means "no opinion" (a huge speed that never wins). While intercepting we also report a
    # gentle braking value for the planner's bookkeeping.
    self.output_v_target = self.v_target if self.is_active else V_CRUISE_UNSET
    self.output_a_target = -A_COMFORT if self.state == VisionLeadState.intercepting else 0.
