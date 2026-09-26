"""Populate the registries.

Importing this module is what makes ``type: identity`` or ``type: sim_watchdog``
resolvable. Without it, registration is a side effect of whichever module a
caller happened to import first -- which means a config file that works in one
entry point fails in another.

Implementations needing only the base dependencies are imported here. Anything
pulling a simulator, a learning framework or a hardware SDK is *declared*
against a module path and imported on first use, so that
``import teleop_sim.core.config`` never drags in MuJoCo or a serial driver.
"""

from __future__ import annotations

from teleop_sim.control import sources as _sources  # noqa: F401
from teleop_sim.control.safety import contact as _contact  # noqa: F401
from teleop_sim.control.safety import hardware_safety as _hardware_safety  # noqa: F401
from teleop_sim.control.safety import sim_watchdog as _sim_watchdog  # noqa: F401
from teleop_sim.control.safety import vision_hazard as _vision_hazard  # noqa: F401
from teleop_sim.core.registry import POLICIES, ROBOTS
from teleop_sim.policies import constant as _constant  # noqa: F401
from teleop_sim.reset import none as _reset_none  # noqa: F401
from teleop_sim.retarget import identity as _identity  # noqa: F401
from teleop_sim.retarget import ik as _ik  # noqa: F401
from teleop_sim.success import hardware as _success_hardware  # noqa: F401
from teleop_sim.success import never as _success_never  # noqa: F401

# Declared, not imported: `import teleop_sim.core.config` must not drag in
# MuJoCo. The module is imported the first time a config names this type.
ROBOTS.declare("mujoco", "teleop_sim.robots.sim.mujoco_robot")
# A running robots_realtime session (e.g. gem13), over its ZMQ bus.
ROBOTS.declare("robots_realtime", "teleop_sim.robots.real.rr_bridge")
# VLM-guided pick / lift / place. Plans with MuJoCo kinematics, so declared.
POLICIES.declare("pick_lift_place", "teleop_sim.tasks.pick_lift_place")
POLICIES.declare("linear_sweep", "teleop_sim.tasks.linear_sweep")

# Further heavyweight backends are declared the same way as they arrive:
#
#   ROBOTS.declare("feetech", "teleop_sim.robots.real.feetech")
#   TELEOPS.declare("leader_arm", "teleop_sim.teleop.leader_arm")
#   SUCCESS.declare("sim_state", "teleop_sim.success.sim_state")
