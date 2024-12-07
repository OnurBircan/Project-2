import time
import numpy as np
import heapq

import mujoco
import mujoco.viewer
import pickle

from simple_pid import PID




class Sensor:
  def __init__(self, noise_std=0.1):
    self.noise_std = noise_std

  def measure(self, true_value):
    noise = np.random.normal(0, self.noise_std)
    return true_value + noise


class KalmanFilter:
  def __init__(self, process_variance, measurement_variance, initial_estimate=0, initial_uncertainty=1):
    self.estimate = initial_estimate
    self.uncertainty = initial_uncertainty
    self.process_variance = process_variance
    self.measurement_variance = measurement_variance

  def predict(self, control_input=0):
    self.estimate += control_input
    self.uncertainty += self.process_variance

  def update(self, measurement):
    kalman_gain = self.uncertainty / (self.uncertainty + self.measurement_variance)
    self.estimate = self.estimate + kalman_gain * (measurement - self.estimate)
    self.uncertainty = (1 - kalman_gain) * self.uncertainty

    return self.estimate
# Örnek:
sensor = Sensor(noise_std=0.5)
kalman_filter = KalmanFilter(process_variance=0.1, measurement_variance=0.5)

true_value = 10  # Gerçek sistem durumu
sensor_measurement = sensor.measure(true_value)
kalman_filter.predict()
estimated_value = kalman_filter.update(sensor_measurement)

print(f"Gerçek Değer: {true_value}, Sensör Ölçümü: {sensor_measurement}, Tahmin Edilen Değer: {estimated_value}")


def save_data(filename, positions, velocities):
    data = {'positions': positions, 'velocities': velocities}
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def pid_to_thrust(input: np.array):
  """ Maps controller output to manipulated variable.

  Args:
      input (np.array): w € [3x1]

  Returns:
      np.array: [3x4]
  """
  c_to_F =np.array([
      [-0.25, 0.25, 0.25, -0.25],
      [0.25, 0.25, -0.25, -0.25],
      [-0.25, 0.25, -0.25, 0.25]
  ]).transpose()

  return np.dot((c_to_F*input),np.array([1,1,1]))

def outer_pid_to_thrust(input: np.array):
  """ Maps controller output to manipulated variable.

  Args:
      input (np.array): w € [3x1]

  Returns:
      np.array: [3x4]
  """
  c_to_F =np.array([
      [0.25, 0.25, -0.25, -0.25],
      [0.25, -0.25, -0.25, 0.25],
      [0.25, 0.25, 0.25, 0.25]
  ]).transpose()

  return np.dot((c_to_F*input),np.array([1,1,1]))

class PDController:
  def __init__(self, kp, kd, setpoint):
    self.kp = kp
    self.kd = kd
    self.setpoint = setpoint
    self.prev_error = 0

  def compute(self, measured_value):
    error = self.setpoint - measured_value
    derivative = error - self.prev_error
    output = (self.kp * error) + (self.kd * derivative)
    self.prev_error = error
    return output


class ImprovedPIDController:
  def __init__(self, kp, ki, kd, setpoint, output_limits=(-float('inf'), float('inf'))):
    self.kp = kp
    self.ki = ki
    self.kd = kd
    self.setpoint = setpoint
    self.prev_error = 0
    self.integral = 0
    self.output_limits = output_limits

  def compute(self, measured_value):
    error = self.setpoint - measured_value
    self.integral += error
    derivative = error - self.prev_error

    # Anti-windup: Integral term constraint
    self.integral = max(self.output_limits[0], min(self.integral, self.output_limits[1]))

    output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
    output = max(self.output_limits[0], min(output, self.output_limits[1]))

    self.prev_error = error
    return output

  def adapt_gain(self, factor):
    """Dynamically adjust PID gains based on a scaling factor."""
    self.kp *= factor
    self.ki *= factor
    self.kd *= factor

  import heapq


class RealTrajectoryPlanner:
  def __init__(self, target, obstacles, vel_limit=2):
    self.target = target
    self.obstacles = obstacles
    self.vel_limit = vel_limit

  def is_in_collision(self, point):
    for obs in self.obstacles:
      dist = np.linalg.norm(np.array(point[:2]) - np.array(obs[:2]))
      if dist <= obs[2]:  # Check if point is within obstacle radius
        return True
    return False

  def astar(self):
    open_set = []
    heapq.heappush(open_set, (0, self.start))
    came_from = {}
    cost_so_far = {tuple(self.start): 0}

    while open_set:
      _, current = heapq.heappop(open_set)
      if np.linalg.norm(np.array(current) - np.array(self.goal)) < 0.1:
        break

      for dx, dy in [(0.1, 0), (0, 0.1), (-0.1, 0), (0, -0.1)]:
        neighbor = (current[0] + dx, current[1] + dy)
        if self.is_in_collision(neighbor):
          continue

        new_cost = cost_so_far[tuple(current)] + np.linalg.norm(np.array(neighbor) - np.array(current))
        if tuple(neighbor) not in cost_so_far or new_cost < cost_so_far[tuple(neighbor)]:
          cost_so_far[tuple(neighbor)] = new_cost
          priority = new_cost + np.linalg.norm(np.array(self.goal) - np.array(neighbor))
          heapq.heappush(open_set, (priority, neighbor))
          came_from[tuple(neighbor)] = current

    self.reconstruct_path(came_from)

  def reconstruct_path(self, came_from):
    current = tuple(self.goal)
    path = [self.goal]
    while current in came_from:
      current = came_from[current]
      path.append(current)
    path.reverse()
    self.path = path

  def get_next_waypoint(self, current_position):
    for waypoint in self.path:
      if np.linalg.norm(np.array(current_position) - np.array(waypoint)) < 0.1:
        continue
      return waypoint
    return self.goal

  """Generate Path from 1 point directly to another"""

  def __init__(self, target, vel_limit = 2) -> None:
    # TODO: MPC
    self.target = target  
    self.vel_limit = vel_limit
    # setpoint target location, controller output: desired velocity.
    self.pid_x = PID(2, 0.15, 1.5, setpoint = self.target[0],
                output_limits = (-vel_limit, vel_limit),)
    self.pid_y = PID(2, 0.15, 1.5, setpoint = self.target[1],
                output_limits = (-vel_limit, vel_limit))
  
  def __call__(self, loc: np.array):
    """Calls planner at timestep to update cmd_vel"""
    velocites = np.array([0,0,0])
    velocites[0] = self.pid_x(loc[0])
    velocites[1] = self.pid_y(loc[1])
    return velocites

  def get_velocities(self,loc: np.array, target: np.array,
                     time_to_target: float = None,
                     flight_speed: float = 0.5) -> np.array:
    """Compute

    Args:
        loc (np.array): Current location in world coordinates.
        target (np.array): Desired location in world coordinates
        time_to_target (float): If set, adpats length of velocity vector.

    Returns:
        np.array: returns velocity vector in world coordinates.
    """

    direction = target - loc
    distance = np.linalg.norm(direction)
    # maps drone velocities to one.
    if distance > 1:
        velocities = flight_speed * direction / distance

    else:
        velocities =  direction * distance

    return velocities

  def get_alt_setpoint(self, loc: np.array) -> float:

    target = self.target
    distance = target[2] - loc[2]
    
    # maps drone velocities to one.
    if distance > 0.5:
        time_sample = 1/4
        time_to_target =  distance / self.vel_limit
        number_steps = int(time_to_target/time_sample)
        # compute distance for next update
        delta_alt = distance / number_steps

        # 2 times for smoothing
        alt_set = loc[2] + 2 * delta_alt
    
    else:
        alt_set = target[2]

    return alt_set

  def update_target(self, target):
    """Update targets"""
    self.target = target  
    # setpoint target location, controller output: desired velocity.
    self.pid_x.setpoint = self.target[0]
    self.pid_y.setpoint = self.target[1]


class dummySensor:
  """Dummy sensor data. So the control code remains intact."""

  def __init__(self, d):
    self.position = d.qpos
    self.velocity = d.qvel
    self.acceleration = d.qacc
    self.position_sensor = Sensor(noise_std=0.1)  # Example: Adding noise
    self.velocity_sensor = Sensor(noise_std=0.1)
    self.acceleration_sensor = Sensor(noise_std=0.1)

  def get_position(self):
    # Use the Kalman filter to get a filtered position
    return self.position_sensor.measure(self.position)

  def get_velocity(self):
    # Use the Kalman filter to get a filtered velocity
    return self.velocity_sensor.measure(self.velocity)

  def get_acceleration(self):
    # Use the Kalman filter to get a filtered acceleration
    return self.acceleration_sensor.measure(self.acceleration)


class drone:
  """Simple drone classe."""
  def __init__(self, target=np.array((0,0,0))):
    self.m = mujoco.MjModel.from_xml_path('mujoco_menagerie-main/skydio_x2/scene.xml')
    self.d = mujoco.MjData(self.m)
    start = [0 , 0]
    goal = np.array([0, 0, 1])  # Hedefin (x, y, z) koordinatları
    # Diğer nesne başlatma işlemleri
    obstacles = [(0.5, 0.5, 0.2), (1.5, 1.5, 0.3)]  # Örnek engeller (x, y, yarıçap)
    self.planner = RealTrajectoryPlanner(target=target, obstacles=obstacles)
    self.sensor = dummySensor(self.d)

    # instantiate controllers

    # inner control to stabalize inflight dynamics
    self.pid_alt = PID(5.50844,0.57871, 1.2,setpoint=0,) # PIDController(0.050844,0.000017871, 0, 0) # thrust
    self.pid_roll = PID(2.6785,0.56871, 1.2508, setpoint=0, output_limits = (-1,1) ) #PID(11.0791,2.5263, 0.10513,setpoint=0, output_limits = (-1,1) )
    self.pid_pitch = PID(2.6785,0.56871, 1.2508, setpoint=0, output_limits = (-1,1) )
    self.pid_yaw =  PID(0.54, 0, 5.358333, setpoint=1, output_limits = (-3,3) )# PID(0.11046, 0.0, 15.8333, setpoint=1, output_limits = (-2,2) )

    # outer control loops
    self.pid_v_x = PID(0.1, 0.003, 0.02, setpoint = 0,
                output_limits = (-0.1, 0.1))
    self.pid_v_y = PID(0.1, 0.003, 0.02, setpoint = 0,
                  output_limits = (-0.1, 0.1))

  def update_outer_control(self):
    """Updates outer control loop for trajectory planning"""
    v = self.sensor.get_velocity()
    location = self.sensor.get_position()[:3]

    # Get next waypoint using the planner
    next_waypoint = self.planner.get_next_waypoint(location)

    # Compute velocities towards the next waypoint
    velocites = self.planner.get_velocities(loc=location, target=next_waypoint)

    # Update setpoints for the PID controllers
    self.pid_alt.setpoint = self.planner.get_alt_setpoint(location)
    self.pid_v_x.setpoint = velocites[0]
    self.pid_v_y.setpoint = velocites[1]

    # Compute angles for inner control
    angle_pitch = self.pid_v_x(v[0])
    angle_roll = -self.pid_v_y(v[1])
    self.pid_pitch.setpoint = angle_pitch
    self.pid_roll.setpoint = angle_roll

  def update_inner_control(self):
    """Upates inner control loop and sets actuators to stabilize flight
    dynamics"""
    alt = self.sensor.get_position()[2]
    angles = self.sensor.get_position()[3:] # roll, yaw, pitch
    
    # apply PID
    cmd_thrust = self.pid_alt(alt) + 3.2495
    cmd_roll = - self.pid_roll(angles[1])
    cmd_pitch = self.pid_pitch(angles[2])
    cmd_yaw = - self.pid_yaw(angles[0])

    #transfer to motor control
    out = self.compute_motor_control(cmd_thrust, cmd_roll, cmd_pitch, cmd_yaw)
    self.d.ctrl[:4] = out

  #  as the drone is underactuated we set
  def compute_motor_control(self, thrust, roll, pitch, yaw):
    motor_control = [
      thrust + roll + pitch - yaw,
      thrust - roll + pitch + yaw,
      thrust - roll -  pitch - yaw,
      thrust + roll - pitch + yaw
    ]
    return motor_control

# -------------------------- Initialization ----------------------------------
my_drone = drone(target=np.array((0,0,1)))

with mujoco.viewer.launch_passive(my_drone.m, my_drone.d) as viewer:
  time.sleep(5)
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  step = 1

  while viewer.is_running() and time.time() - start < 30:
    step_start = time.time()
    
    # flight program
    if time.time()- start > 2:
      my_drone.planner.update_target(np.array((1,1,1)))

    if time.time()- start > 10:
      my_drone.planner.update_target(np.array((-1,1,2)))

    if time.time()- start > 18:
      my_drone.planner.update_target(np.array((-1,-1,0.5)))

    # outer control loop
    if step % 20 == 0:
     my_drone.update_outer_control()
    # Inner control loop
    my_drone.update_inner_control()

    mujoco.mj_step(my_drone.m, my_drone.d)

    # Example modification of a viewer option: toggle contact points every two seconds.
    with viewer.lock():
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(my_drone.d.time % 2)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()
    
    # Increment to time slower outer control loop
    step += 1
    
    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = my_drone.m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)
