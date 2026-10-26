#!/usr/bin/env python3
"""
Central controller - attachment model dynamics and action selection.

The mode is set with roslaunch ... controller:=<mode> (param /attachment/controller):
  decoupled  (default) each robot integrates its own oscillator and an internal model of the
             partner; the coupling networks get the need gap of that internal model and silent
             vocal inputs. Both robots use the caregiver networks.
  coupled    one integrator holds both robots' needs (CentralAttachmentController).
  observed   as decoupled, but the coupling networks get what the robot senses: the distance to
             the partner (ground truth or odometry, param /attachment/distance) and the vocal
             exchange on the simulated audio topics. Each robot uses its own networks.
  baseline   as decoupled, but only the robot's own half of its internal dyad uses the learned
             coupling; the partner model follows the standard (coupled) model with the analytic
             coupling, as a baseline. baseline-observed: the same, with the networks fed what the
             robot senses (as in observed).
  notebook   no partner model: each robot integrates only its own need with the equations of the
             agent simulator the networks were trained in (attachment_with_nn.ipynb), coupled
             through the networks fed with what it senses.
Every controller saves its internal state and coupling inputs/outputs per tick to
controller_<name>.npy in the run folder.
"""
import sys
sys.path.insert(0, '/workspace/.venv/lib/python3.8/site-packages')
import rospy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import threading
from collections import deque
from std_msgs.msg import Bool
from gazebo_msgs.msg import ModelStates
from attachment_model.msg import Action
from subscribers import EmotionSubscriber, OdomSubscriber
from utils import ARENA_SIZE, SPAWN_POSE, initial_state, odom_to_world, latest_experiment_dir, fix_owner

# Training arena of the coupling networks: [-1, 1]^2, side 2 (attachment_with_nn.ipynb)
TRAINING_ARENA = 2.0
# Distance filter used to build the physical-network training inputs (filter_dist in the notebook):
# a causal conv1d with an exponential kernel. conv1d is a cross-correlation, so the largest weight
# falls on the oldest sample of the 20-step window; this is reproduced as trained.
FILTER_TAPS = 20
FILTER_KERNEL = np.exp(-np.arange(FILTER_TAPS) / 10.0)
FILTER_KERNEL /= FILTER_KERNEL.sum()
# Call and evidence traces of the agent simulator
KP, KE, KAPPA = 0.1, 0.1, 0.2


# Neural Network Models
class PhysicalDistanceNN(nn.Module):
    def __init__(self, D_in):
        super(PhysicalDistanceNN, self).__init__()
        hidden_dim = 32
        self.fc1 = nn.Linear(D_in, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2*hidden_dim)
        self.fc3 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.fc4(x)
        return x

class EmotionalDistanceNN(nn.Module):
    def __init__(self, D_in):
        super(EmotionalDistanceNN, self).__init__()
        hidden_dim = 64
        self.fc1 = nn.Linear(D_in, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2*hidden_dim)
        self.fc3 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))
        x = F.silu(self.fc3(x))
        x = self.fc4(x)
        return x


def load_networks(device, agent):
    """The physical and emotional coupling networks trained for `agent` ('parent' or 'child')."""
    model_dir = os.path.dirname(__file__)
    physical = PhysicalDistanceNN(D_in=4).to(device)
    emotional = EmotionalDistanceNN(D_in=6).to(device)
    physical.load_state_dict(torch.load(os.path.join(model_dir, f'physical_{agent}_coupling_model.pth'),
                                        map_location=device))
    emotional.load_state_dict(torch.load(os.path.join(model_dir, f'emotional_{agent}_coupling_model.pth'),
                                         map_location=device))
    physical.eval()
    emotional.eval()
    return physical, emotional


class TickLog:
    """Per-tick record of a controller, saved to controller_<name>.npy on shutdown."""

    def __init__(self, name, meta):
        self.name = name
        self.rows = {}
        self.meta = meta
        rospy.on_shutdown(self.save)

    def add(self, **values):
        for key, value in values.items():
            self.rows.setdefault(key, []).append(value)

    def save(self):
        data = {key: np.asarray(value, dtype=float) for key, value in self.rows.items()}
        data['meta'] = self.meta
        filename = os.path.join(latest_experiment_dir(), f"controller_{self.name}.npy")
        np.save(filename, data, allow_pickle=True)
        fix_owner(filename)
        rospy.loginfo(f"Saved controller log to {filename}")


class _GroundTruth:
    """One /gazebo/model_states subscription shared by every controller of this process."""
    _instance = None

    def __init__(self):
        self.pose = {}
        rospy.Subscriber("/gazebo/model_states", ModelStates, self._callback, queue_size=1)

    @classmethod
    def shared(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _callback(self, msg):
        for name in ("child", "parent"):
            if name in msg.name:
                position = msg.pose[msg.name.index(name)].position
                self.pose[name] = (position.x, position.y)


class Observations:
    """What one robot senses about its partner: distance and the vocal exchange.

    The inputs are built as in the agent simulator the networks were trained in: the distance in
    training-arena units through the training filter, and leaky call and evidence traces updated
    once per controller tick (0.01 model time, the simulator's step).
    """

    def __init__(self, robot_name, source):
        self.robot_name = robot_name
        self.partner = "parent" if robot_name == "child" else "child"
        self.source = source
        if source == "gt":
            self.ground_truth = _GroundTruth.shared()
        elif source == "odom":
            self.own_pose = OdomSubscriber(robot_name)
            self.partner_pose = OdomSubscriber(self.partner)
        else:
            raise ValueError(f"distance source must be gt or odom, got {source!r}")
        self.own_call = False
        self.partner_call = False
        rospy.Subscriber(f"/{robot_name}/audio_simulation", Bool, self._own_cb)
        rospy.Subscriber(f"/{self.partner}/audio_simulation", Bool, self._partner_cb)
        x0, y0, _ = SPAWN_POSE[robot_name]
        x1, y1, _ = SPAWN_POSE[self.partner]
        self.last_distance = float(np.hypot(x0 - x1, y0 - y1))
        self.history = deque([0.0] * (FILTER_TAPS - 1), maxlen=FILTER_TAPS)   # zero padding as in training
        self.call_trace = 0.0
        self.evidence = 0.0

    def _own_cb(self, msg):
        self.own_call = bool(msg.data)

    def _partner_cb(self, msg):
        self.partner_call = bool(msg.data)

    def _position(self, name):
        if self.source == "gt":
            return self.ground_truth.pose.get(name, (float("nan"), float("nan")))
        sub = self.own_pose if name == self.robot_name else self.partner_pose
        x, y, _ = odom_to_world(name, sub.posx, sub.posy, sub.yaw)
        return x, y

    def distance_m(self):
        ox, oy = self._position(self.robot_name)
        px, py = self._position(self.partner)
        d = float(np.hypot(ox - px, oy - py))
        if np.isfinite(d):
            self.last_distance = d
        return self.last_distance

    def step(self):
        """Advance the traces by one tick and return the network inputs."""
        raw = self.distance_m()
        self.history.append(raw * TRAINING_ARENA / ARENA_SIZE)
        filtered = float(np.dot(FILTER_KERNEL, np.asarray(self.history)))
        own, partner = float(self.own_call), float(self.partner_call)
        previous_call_trace = self.call_trace
        self.evidence = (1 - KE) * self.evidence + KAPPA * partner * previous_call_trace
        self.call_trace = (1 - KP) * previous_call_trace + KAPPA * own
        return dict(distance_m=raw, distance_in=filtered, evidence=self.evidence,
                    call_trace=self.call_trace, own_call=own, partner_call=partner)


class CentralAttachmentController:
    """Central controller implementing attachment dynamics."""

    def __init__(self, initial=None):
        # Publishers
        self.controller_publisher = rospy.Publisher(f'/central/controller', Action, queue_size=0)

        # Subscribers
        self.emotion = EmotionSubscriber()

        # Set variables for the node
        self.time = 0
        initial = initial or (-1.0, 1.0, -1.0, 1.0)
        self.child_need, self.child_accumulated_need, self.parent_need, self.parent_accumulated_need = initial
        self.k0 = 0.8
        self.a = 1.0
        self.c = 1.0
        self.b = 0.7
        self.delta = 0.1
        self.physical_distance = 0.0
        self.emotional_distance = 0.0
        self.epsilonAv = self.emotion.avoidant
        self.epsilonAm = self.emotion.ambivalent
        self.h = 0.01

        # Action message initialization
        self.action_msg = Action()
        self.action_msg.child = 0
        self.action_msg.parent = 0
        self.action_msg.emotional_distance = 0
        self.action_msg.physical_distance = 0
        self.action_msg.child_need = 0
        self.action_msg.parent_need = 0
        self.action_msg.child_accumulated_need = 0
        self.action_msg.parent_accumulated_need = 0

        self.log = TickLog("central", dict(mode="coupled", initial=list(initial)))

    def _calculate_physical_distance(self):
        """Calculate physical distance based on model"""
        return (self.child_accumulated_need + self.parent_accumulated_need)/2 + self.child_accumulated_need

    def _calculate_emotional_distance(self):
        """Calculate emotional distance based on model"""
        return self.k0*((self.child_accumulated_need - self.parent_accumulated_need)/2 + self.child_accumulated_need) + self.delta*self.k0

    def run(self):
        """Main control loop implementing attachment dynamics."""
        rate = rospy.Rate(30)
        A_explore = lambda x: np.heaviside(-x,0)

        while not rospy.is_shutdown():
            self.time += self.h

            # Calculate physical distance (normalized to [0,1])
            self.physical_distance = self._calculate_physical_distance()

            # Calculate emotional distance
            self.emotional_distance = self._calculate_emotional_distance()

            calculated_need_accumulation = [self.child_need, self.child_accumulated_need, self.parent_need, self.parent_accumulated_need]

            # Calculate needs and accumulated needs, Runga-Kutta 4th order
            k1 = self.f(self.time, calculated_need_accumulation)
            k2 = self.f(self.time + self.h/2.0, calculated_need_accumulation + self.h*k1/2.0)
            k3 = self.f(self.time + self.h/2.0, calculated_need_accumulation + self.h*k2/2.0)
            k4 = self.f(self.time + self.h, calculated_need_accumulation + self.h*k3)
            calculated_need_accumulation = calculated_need_accumulation + self.h*(k1 + 2*k2 + 2*k3 + k4)/6.0

            # Update action message
            self.action_msg.child = int(A_explore(calculated_need_accumulation[0]))
            self.action_msg.parent = int(A_explore(calculated_need_accumulation[2]))
            self.action_msg.physical_distance = self.physical_distance
            self.action_msg.emotional_distance = self.emotional_distance
            self.action_msg.child_need = calculated_need_accumulation[0]
            self.action_msg.parent_need = calculated_need_accumulation[2]
            self.action_msg.child_accumulated_need = calculated_need_accumulation[1]
            self.action_msg.parent_accumulated_need = calculated_need_accumulation[3]
            self.controller_publisher.publish(self.action_msg)

            # Update model variables
            [self.child_need,self.child_accumulated_need,self.parent_need,self.parent_accumulated_need] = calculated_need_accumulation
            self.log.add(t=rospy.get_time(), state=list(calculated_need_accumulation),
                         physical=self.physical_distance, emotional=self.emotional_distance)
            rate.sleep()

    # Model
    def f(self, t, r):
        x1 = r[0]
        y1 = r[1]
        x2 = r[2]
        y2 = r[3]
        ym = (y1 + y2)/2.0
        xm = (x1 + x2)/2.0
        # Calculate needs and accumulated needs
        dx1 = -self.a*(4.0*self.c*x1**3 - 2.0*x1) - y1
        dy1 = self.b*x1- self.epsilonAm*(y2 + y1 - self.physical_distance)  - self.epsilonAv*(y2 - y1 - self.emotional_distance)
        dx2 = -self.a*(4.0*self.c*x2**3 - 2.0*x2) - y2
        dy2 = self.b*x2  - self.epsilonAm*(y2 + y1 + self.physical_distance)  - self.epsilonAv*(y2 - y1 - self.emotional_distance)
        return np.array([dx1, dy1, dx2, dy2])

class IndividualAttachmentController:
    """Individual controller implementing attachment dynamics.

    The robot integrates its own oscillator and an internal model of its partner. mode='decoupled'
    feeds the coupling networks the need gap of the internal model and silent vocal inputs;
    mode='observed' feeds them what the robot senses (Observations). In the 'baseline' modes the
    partner model is integrated with the analytic coupling of the standard model instead of the
    learned one ('baseline': network inputs as decoupled, 'baseline-observed': as observed).
    """

    def __init__(self, robot_name, mode="decoupled", initial=None, distance_source="gt"):
        # Don't initialize node here - will be done in main

        # Publishers
        self.controller_publisher = rospy.Publisher(f'/{robot_name}/controller', Action, queue_size=10)

        # Subscribers
        self.emotion = EmotionSubscriber()

        # Store robot name
        self.robot_name = robot_name
        self.mode = mode

        # Set variables for the node
        self.time = 0
        initial = initial or (-1.0, 1.0, -1.0, 1.0)
        self.child_need, self.child_accumulated_need, self.parent_need, self.parent_accumulated_need = initial
        self.k0 = 0.8
        self.a = 1.0
        self.c = 1.0
        self.b = 0.7
        self.delta = 0.1
        self.physical_distance = 0.0
        self.emotional_distance = 0.0
        self.epsilonAv = self.emotion.avoidant
        self.epsilonAm = self.emotion.ambivalent
        self.h = 0.01

        # Action message initialization
        self.action_msg = Action()
        self.action_msg.child = 0
        self.action_msg.parent = 0
        self.action_msg.emotional_distance = 0
        self.action_msg.physical_distance = 0
        self.action_msg.child_need = 0
        self.action_msg.parent_need = 0
        self.action_msg.child_accumulated_need = 0
        self.action_msg.parent_accumulated_need = 0

        # Load neural network models: the caregiver's for both robots in the decoupled mode (as
        # recorded so far), the robot's own when it uses its own observations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net_agent = "parent" if mode in ("decoupled", "baseline") else robot_name
        self.sensing = mode in ("observed", "baseline-observed")
        self.partner_baseline = mode.startswith("baseline")
        self.partner_physical = self.partner_emotional = 0.0
        try:
            self.physical_NN, self.emotional_NN = load_networks(self.device, self.net_agent)
            rospy.loginfo(f"Successfully loaded neural network models ({self.net_agent})")
        except Exception as e:
            rospy.logwarn(f"Failed to load neural network models: {e}. Using fallback calculations.")
            self.physical_NN = None
            self.emotional_NN = None

        self.observations = Observations(robot_name, distance_source) if self.sensing else None
        self.obs = {}
        self.log = TickLog(robot_name, dict(mode=mode, networks=self.net_agent, initial=list(initial),
                                            distance=distance_source if self.sensing else None))

    def _own_state(self):
        if self.robot_name == "child":
            return self.child_need, self.child_accumulated_need
        return self.parent_need, self.parent_accumulated_need

    def _physical_input(self):
        """Input of the physical network: [distance, x, y, ambivalence]."""
        need, accumulated_need = self._own_state()
        if self.sensing:
            distance = self.obs["distance_in"]
        else:
            # Use euclidean distance between accumulated needs
            distance = np.sqrt((self.child_accumulated_need - self.parent_accumulated_need)**2)
        return [distance, need, accumulated_need, self.epsilonAm]

    def _emotional_input(self):
        """Input of the emotional network: [evidence_trace, other_vocal, self_vocal, x, y, avoidance]."""
        need, accumulated_need = self._own_state()
        if self.sensing:
            return [self.obs["evidence"], self.obs["partner_call"], self.obs["own_call"],
                    need, accumulated_need, self.epsilonAv]
        # For simplicity, use zeros for vocalization tracking
        return [0.0,  # evidence_trace (could track this over time if needed)
                0,    # other vocalization
                0,    # self vocalization
                need,
                accumulated_need,
                self.epsilonAv]

    def _calculate_physical_distance(self):
        """Calculate physical distance using neural network or fallback to model"""
        if self.physical_NN is not None:
            try:
                input_tensor = torch.tensor(self._physical_input(), dtype=torch.float32).to(self.device)

                with torch.no_grad():
                    coupling = self.physical_NN(input_tensor).item()

                return coupling
            except Exception as e:
                rospy.logwarn_throttle(10, f"Error in physical NN: {e}. Using fallback.")

        # Fallback to original calculation
        return (self.child_accumulated_need + self.parent_accumulated_need)/2 + self.child_accumulated_need

    def _calculate_emotional_distance(self):
        """Calculate emotional distance using neural network or fallback to model"""
        if self.emotional_NN is not None:
            try:
                input_tensor = torch.tensor(self._emotional_input(), dtype=torch.float32).to(self.device)

                with torch.no_grad():
                    coupling = self.emotional_NN(input_tensor).item()

                return coupling
            except Exception as e:
                rospy.logwarn_throttle(10, f"Error in emotional NN: {e}. Using fallback.")

        # Fallback to original calculation
        return self.k0*((self.child_accumulated_need - self.parent_accumulated_need)/2 + self.child_accumulated_need) + self.delta*self.k0

    def run(self):
        """Main control loop implementing attachment dynamics."""
        rate = rospy.Rate(30)
        A_explore = lambda x: np.heaviside(-x,0)

        while not rospy.is_shutdown():
            self.time += self.h

            if self.observations is not None:
                self.obs = self.observations.step()

            # Calculate physical distance (normalized to [0,1])
            self.physical_distance = self._calculate_physical_distance()

            # Calculate emotional distance
            self.emotional_distance = self._calculate_emotional_distance()

            if self.partner_baseline:
                # analytic coupling of the standard model for the partner model, held over the tick
                self.partner_physical = (self.child_accumulated_need + self.parent_accumulated_need)/2 + self.child_accumulated_need
                self.partner_emotional = self.k0*((self.child_accumulated_need - self.parent_accumulated_need)/2 + self.child_accumulated_need) + self.delta*self.k0

            calculated_need_accumulation = [self.child_need, self.child_accumulated_need, self.parent_need, self.parent_accumulated_need]

            # Calculate needs and accumulated needs, Runga-Kutta 4th order
            k1 = self.f(self.time, calculated_need_accumulation)
            k2 = self.f(self.time + self.h/2.0, calculated_need_accumulation + self.h*k1/2.0)
            k3 = self.f(self.time + self.h/2.0, calculated_need_accumulation + self.h*k2/2.0)
            k4 = self.f(self.time + self.h, calculated_need_accumulation + self.h*k3)
            calculated_need_accumulation = calculated_need_accumulation + self.h*(k1 + 2*k2 + 2*k3 + k4)/6.0

            # Update action message
            self.action_msg.child = int(A_explore(calculated_need_accumulation[0]))
            self.action_msg.parent = int(A_explore(calculated_need_accumulation[2]))
            self.action_msg.physical_distance = self.physical_distance
            self.action_msg.emotional_distance = self.emotional_distance
            self.action_msg.child_need = calculated_need_accumulation[0]
            self.action_msg.parent_need = calculated_need_accumulation[2]
            self.action_msg.child_accumulated_need = calculated_need_accumulation[1]
            self.action_msg.parent_accumulated_need = calculated_need_accumulation[3]
            self.controller_publisher.publish(self.action_msg)

            # Update model variables
            [self.child_need,self.child_accumulated_need,self.parent_need,self.parent_accumulated_need] = calculated_need_accumulation
            self.log.add(t=rospy.get_time(), state=list(calculated_need_accumulation),
                         physical=self.physical_distance, emotional=self.emotional_distance,
                         **({"partner_physical": self.partner_physical, "partner_emotional": self.partner_emotional}
                            if self.partner_baseline else {}),
                         **{key: value for key, value in self.obs.items()})
            rate.sleep()

    # Model
    def f(self, t, r):
        x1 = r[0]
        y1 = r[1]
        x2 = r[2]
        y2 = r[3]
        ym = (y1 + y2)/2.0
        xm = (x1 + x2)/2.0
        # Calculate needs and accumulated needs
        # coupling of the child half (1) and the caregiver half (2): learned for both, or, with the
        # baseline partner model, learned for the robot's own half and analytic for the partner's
        dp1 = dp2 = self.physical_distance
        de1 = de2 = self.emotional_distance
        if self.partner_baseline:
            if self.robot_name == "child":
                dp2, de2 = self.partner_physical, self.partner_emotional
            else:
                dp1, de1 = self.partner_physical, self.partner_emotional
        dx1 = -self.a*(4.0*self.c*x1**3 - 2.0*x1) - y1
        dy1 = self.b*x1- self.epsilonAm*(y2 + y1 - dp1)  - self.epsilonAv*(y2 - y1 - de1)
        dx2 = -self.a*(4.0*self.c*x2**3 - 2.0*x2) - y2
        dy2 = self.b*x2  - self.epsilonAm*(y2 + y1 + dp2)  - self.epsilonAv*(y2 - y1 - de2)
        return np.array([dx1, dy1, dx2, dy2])


class NotebookAgentController:
    """One robot with no model of its partner, as in the agent simulator the networks were trained in.

    The robot integrates only its own need with the simulator's equations (Euler, dt = 0.01,
    b = 0.5, dx = x - 2x^3 - y) and takes the whole coupling term from its own networks, fed with
    what it senses (Observations): dy = b x -/+ ambivalence * Phi_p - avoidance * Phi_e
    (minus for the child, plus for the caregiver, as in Child.step and Parent.step).
    """

    def __init__(self, robot_name, initial=None, distance_source="gt"):
        self.robot_name = robot_name
        self.controller_publisher = rospy.Publisher(f'/{robot_name}/controller', Action, queue_size=10)
        self.emotion = EmotionSubscriber()
        self.epsilonAv = self.emotion.avoidant
        self.epsilonAm = self.emotion.ambivalent
        self.dt = 0.01
        self.b = 0.5
        initial = initial or (-1.0, 1.0, -1.0, 1.0)
        self.x, self.y = (initial[0], initial[1]) if robot_name == "child" else (initial[2], initial[3])
        self.sign = -1.0 if robot_name == "child" else 1.0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.physical_NN, self.emotional_NN = load_networks(self.device, robot_name)
        self.observations = Observations(robot_name, distance_source)
        self.action_msg = Action()
        self.log = TickLog(robot_name, dict(mode="notebook", networks=robot_name, initial=list(initial),
                                            distance=distance_source))

    def run(self):
        rate = rospy.Rate(30)
        A_explore = lambda x: np.heaviside(-x, 0)
        while not rospy.is_shutdown():
            obs = self.observations.step()
            with torch.no_grad():
                physical = self.physical_NN(torch.tensor(
                    [obs["distance_in"], self.x, self.y, self.epsilonAm], dtype=torch.float32).to(self.device)).item()
                emotional = self.emotional_NN(torch.tensor(
                    [obs["evidence"], obs["partner_call"], obs["own_call"], self.x, self.y, self.epsilonAv],
                    dtype=torch.float32).to(self.device)).item()
            self.x = self.x + self.dt * (self.x - 2 * self.x ** 3 - self.y)
            self.y = self.y + self.dt * (self.b * self.x + self.sign * self.epsilonAm * physical
                                         - self.epsilonAv * emotional)

            action = int(A_explore(self.x))
            msg = self.action_msg
            msg.physical_distance, msg.emotional_distance = physical, emotional
            if self.robot_name == "child":
                msg.child, msg.child_need, msg.child_accumulated_need = action, self.x, self.y
                msg.parent, msg.parent_need, msg.parent_accumulated_need = 0, float("nan"), float("nan")
            else:
                msg.parent, msg.parent_need, msg.parent_accumulated_need = action, self.x, self.y
                msg.child, msg.child_need, msg.child_accumulated_need = 0, float("nan"), float("nan")
            self.controller_publisher.publish(msg)
            self.log.add(t=rospy.get_time(), state=[self.x, self.y], physical=physical, emotional=emotional,
                         **obs)
            rate.sleep()


def run_threads(controllers):
    """Run controllers in parallel threads."""
    threads = [threading.Thread(target=c.run) for c in controllers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


if __name__ == '__main__':
    try:
        rospy.init_node('individual_attachment_controllers')
        mode = rospy.get_param("/attachment/controller", "decoupled")
        distance_source = rospy.get_param("/attachment/distance", "gt")
        initial = initial_state(rospy.get_param("/attachment/initial", ""), rospy.get_param("/attachment/seed", -1))
        rospy.set_param("/attachment/initial_state", list(initial))
        rospy.loginfo(f"Attachment controller: mode={mode}, distance={distance_source}, initial={initial}")

        if mode == "coupled":
            CentralAttachmentController(initial).run()
        elif mode in ("decoupled", "observed", "baseline", "baseline-observed"):
            run_threads([IndividualAttachmentController(name, mode, initial, distance_source)
                         for name in ("child", "parent")])
        elif mode == "notebook":
            run_threads([NotebookAgentController(name, initial, distance_source) for name in ("child", "parent")])
        else:
            raise ValueError(f"unknown controller mode {mode!r}")
    except rospy.ROSInterruptException:
        pass
