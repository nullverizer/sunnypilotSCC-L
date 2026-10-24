"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from types import SimpleNamespace

from openpilot.common.constants import CV
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET
from openpilot.sunnypilot.selfdrive.controls.lib.vision_lead_intercept import VisionLeadIntercept, VisionLeadState, \
                                                                             STOCK_TRACKING_DIST

MIN_SET = 20 * CV.MPH_TO_MS
V_SET = 45 * CV.MPH_TO_MS


def make_sm(lead_d=None, lead_v=0., prob=0.9, v_cruise_stock=V_SET):
  leads = [SimpleNamespace(prob=prob, x=[lead_d], v=[lead_v])] if lead_d is not None else []
  return {
    'modelV2': SimpleNamespace(leadsV3=leads),
    'carState': SimpleNamespace(cruiseState=SimpleNamespace(speedCluster=v_cruise_stock)),
  }


class TestVisionLeadIntercept:
  def setup_method(self):
    CP = SimpleNamespace(openpilotLongitudinalControl=False)
    CP_SP = SimpleNamespace(pcmCruiseSpeed=False)  # ICBM active
    self.vli = VisionLeadIntercept(CP, CP_SP)
    self.vli._read_params = lambda: None
    self.vli.enabled = True
    self.vli.is_metric = False

  def step(self, n=1, v_ego=V_SET, v_cruise=V_SET, long_enabled=True, **sm_kwargs):
    for _ in range(n):
      self.vli.update(make_sm(**sm_kwargs), long_enabled, False, v_ego, v_cruise)

  def test_idle_without_lead(self):
    self.step(40)
    assert self.vli.state == VisionLeadState.idle
    assert self.vli.output_v_target == V_CRUISE_UNSET

  def test_disabled_by_toggle_and_op_long(self):
    self.vli.enabled = False
    self.step(40, lead_d=100.)
    assert not self.vli.is_active

    self.vli.enabled = True
    self.vli.CP.openpilotLongitudinalControl = True
    self.step(40, lead_d=100.)
    assert not self.vli.is_active

  def test_ignores_fast_lead_and_close_lead(self):
    self.step(40, lead_d=100., lead_v=15.)
    assert not self.vli.is_active
    self.step(40, lead_d=STOCK_TRACKING_DIST - 5.)
    assert not self.vli.is_active

  def test_debounce_then_smooth_descent(self):
    self.step(5, lead_d=120.)
    assert not self.vli.is_active  # not debounced yet
    self.step(10, lead_d=120.)
    assert self.vli.state == VisionLeadState.intercepting

    # Descending, never increasing, never below the minimum set speed, and rate limited
    prev = self.vli.output_v_target
    d = 120.
    for _ in range(200):
      d -= V_SET * DT_MDL * 0.5
      if d <= STOCK_TRACKING_DIST + 1:
        break
      self.step(lead_d=d)
      assert self.vli.output_v_target <= prev + 1e-6
      assert prev - self.vli.output_v_target <= 1.5 * DT_MDL + 1e-6
      assert self.vli.output_v_target >= MIN_SET - 1e-6
      prev = self.vli.output_v_target
    assert self.vli.output_v_target < V_SET

  def test_holds_at_min_and_releases_when_lead_departs(self):
    self.step(20, lead_d=120.)
    assert self.vli.state == VisionLeadState.intercepting
    # Stock set speed reached the minimum: hold there, do not let ICBM raise it again
    self.step(lead_d=80., v_cruise_stock=MIN_SET)
    assert self.vli.state == VisionLeadState.holding
    self.step(40, lead_d=20., v_ego=0., v_cruise_stock=MIN_SET)
    assert self.vli.state == VisionLeadState.holding
    assert abs(self.vli.output_v_target - MIN_SET) < 0.5

    # Lead drives away: fully release back to the driver's set speed
    self.step(lead_d=30., lead_v=12., v_cruise_stock=MIN_SET)
    assert self.vli.state == VisionLeadState.idle
    assert self.vli.output_v_target == V_CRUISE_UNSET

  def test_releases_to_hold_when_stock_tracking(self):
    self.step(20, lead_d=120.)
    self.step(lead_d=STOCK_TRACKING_DIST - 1., v_cruise_stock=30 * CV.MPH_TO_MS)
    assert self.vli.state == VisionLeadState.holding
    assert abs(self.vli.output_v_target - MIN_SET) < 0.5

  def test_lead_lost_releases(self):
    self.step(20, lead_d=120.)
    assert self.vli.is_active
    self.step(int(1. / DT_MDL))
    assert self.vli.is_active  # brief dropout tolerated
    self.step(int(1.5 / DT_MDL))
    assert not self.vli.is_active

  def test_driver_raising_set_speed_cancels(self):
    self.step(20, lead_d=120.)
    assert self.vli.is_active
    self.step(lead_d=110., v_cruise=V_SET + 1.)
    assert not self.vli.is_active
    self.step(40, lead_d=100., v_cruise=V_SET + 1.)
    assert not self.vli.is_active  # stays cancelled for this lead

  def test_disengage_resets(self):
    self.step(20, lead_d=120.)
    self.step(lead_d=110., long_enabled=False)
    assert not self.vli.is_active
