import time
import numpy as np
import collections
import matplotlib.pyplot as plt
import dm_env

from experiments.robot.aloha.constants import DT, START_ARM_POSE, MASTER_GRIPPER_JOINT_NORMALIZE_FN, PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN
from experiments.robot.aloha.constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN, PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN
from experiments.robot.aloha.constants import PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
# from experiments.robot.aloha.robot_utils import Recorder, ImageRecorder
from experiments.robot.aloha.robot_utils import setup_master_bot, setup_puppet_bot, move_arms, move_grippers

import IPython
e = IPython.embed

import cv2
import argparse
import threading
from collections import deque
import rospy
from std_msgs.msg import Header
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge


# =================================================================================
class ROSArgument:
    publish_rate: int = 40
    arm_steps_length: float = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2]
    use_depth_image: bool = False
    use_robot_base: bool = False
    img_left_topic: str = '/camera_l/color/image_raw'
    img_right_topic: str = '/camera_r/color/image_raw'
    img_front_topic: str = '/camera_f/color/image_raw'
    img_left_depth_topic: str = '/camera_l/depth/image_raw'
    img_right_depth_topic: str = '/camera_r/depth/image_raw'
    img_front_depth_topic: str = '/camera_f/depth/image_raw'
    puppet_arm_left_topic: str = '/puppet/joint_left'
    puppet_arm_right_topic: str = '/puppet/joint_right'
    robot_base_topic: str = '/odom_raw'
    puppet_arm_left_cmd_topic: str = '/master/joint_left'
    puppet_arm_right_cmd_topic: str = '/master/joint_right'
    robot_base_cmd_topic: str = '/cmd_vel'


# [ABOUT ROS] DON'T TOUCH CODE BELOW!
class RosOperator:
    def __init__(self, args):
        self.robot_base_deque = None
        self.puppet_arm_right_deque = None
        self.puppet_arm_left_deque = None
        self.img_front_deque = None
        self.img_right_deque = None
        self.img_left_deque = None
        self.img_front_depth_deque = None
        self.img_right_depth_deque = None
        self.img_left_depth_deque = None
        self.bridge = None
        self.puppet_arm_left_publisher = None
        self.puppet_arm_right_publisher = None
        self.robot_base_publisher = None
        self.puppet_arm_publish_thread = None
        self.puppet_arm_publish_lock = None
        self.args = args
        self.ctrl_state = False
        self.ctrl_state_lock = threading.Lock()
        self.init()
        self.init_ros()

    def init(self):
        self.bridge = CvBridge()
        self.img_left_deque = deque()
        self.img_right_deque = deque()
        self.img_front_deque = deque()
        self.img_left_depth_deque = deque()
        self.img_right_depth_deque = deque()
        self.img_front_depth_deque = deque()
        self.puppet_arm_left_deque = deque()
        self.puppet_arm_right_deque = deque()
        self.robot_base_deque = deque()
        self.puppet_arm_publish_lock = threading.Lock()
        self.puppet_arm_publish_lock.acquire()

    def puppet_arm_publish(self, left, right):
        joint_state_msg = JointState()
        joint_state_msg.header = Header()
        joint_state_msg.header.stamp = rospy.Time.now()  # 设置时间戳
        joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']  # 设置关节名称
        joint_state_msg.position = left
        self.puppet_arm_left_publisher.publish(joint_state_msg)
        joint_state_msg.position = right
        self.puppet_arm_right_publisher.publish(joint_state_msg)

    def robot_base_publish(self, vel):
        vel_msg = Twist()
        vel_msg.linear.x = vel[0]
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = vel[1]
        self.robot_base_publisher.publish(vel_msg)

    def puppet_arm_publish_continuous(self, left, right):
        rate = rospy.Rate(self.args.publish_rate)
        left_arm = None
        right_arm = None
        while True and not rospy.is_shutdown():
            if len(self.puppet_arm_left_deque) != 0:
                left_arm = list(self.puppet_arm_left_deque[-1].position)
            if len(self.puppet_arm_right_deque) != 0:
                right_arm = list(self.puppet_arm_right_deque[-1].position)
            if left_arm is None or right_arm is None:
                rate.sleep()
                continue
            else:
                break
        left_symbol = [1 if left[i] - left_arm[i] > 0 else -1 for i in range(len(left))]
        right_symbol = [1 if right[i] - right_arm[i] > 0 else -1 for i in range(len(right))]
        flag = True
        step = 0
        while flag and not rospy.is_shutdown():
            if self.puppet_arm_publish_lock.acquire(False):
                return
            left_diff = [abs(left[i] - left_arm[i]) for i in range(len(left))]
            right_diff = [abs(right[i] - right_arm[i]) for i in range(len(right))]
            flag = False
            for i in range(len(left)):
                if left_diff[i] < self.args.arm_steps_length[i]:
                    left_arm[i] = left[i]
                else:
                    left_arm[i] += left_symbol[i] * self.args.arm_steps_length[i]
                    flag = True
            for i in range(len(right)):
                if right_diff[i] < self.args.arm_steps_length[i]:
                    right_arm[i] = right[i]
                else:
                    right_arm[i] += right_symbol[i] * self.args.arm_steps_length[i]
                    flag = True
            joint_state_msg = JointState()
            joint_state_msg.header = Header()
            joint_state_msg.header.stamp = rospy.Time.now()  # 设置时间戳
            joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']  # 设置关节名称
            joint_state_msg.position = left_arm
            self.puppet_arm_left_publisher.publish(joint_state_msg)
            joint_state_msg.position = right_arm
            self.puppet_arm_right_publisher.publish(joint_state_msg)
            step += 1
            print("puppet_arm_publish_continuous:", step)
            rate.sleep()

    def puppet_arm_publish_linear(self, left, right):
        num_step = 100
        rate = rospy.Rate(200)

        left_arm = None
        right_arm = None

        while True and not rospy.is_shutdown():
            if len(self.puppet_arm_left_deque) != 0:
                left_arm = list(self.puppet_arm_left_deque[-1].position)
            if len(self.puppet_arm_right_deque) != 0:
                right_arm = list(self.puppet_arm_right_deque[-1].position)
            if left_arm is None or right_arm is None:
                rate.sleep()
                continue
            else:
                break

        traj_left_list = np.linspace(left_arm, left, num_step)
        traj_right_list = np.linspace(right_arm, right, num_step)

        for i in range(len(traj_left_list)):
            traj_left = traj_left_list[i]
            traj_right = traj_right_list[i]
            traj_left[-1] = left[-1]
            traj_right[-1] = right[-1]
            joint_state_msg = JointState()
            joint_state_msg.header = Header()
            joint_state_msg.header.stamp = rospy.Time.now()  # 设置时间戳
            joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']  # 设置关节名称
            joint_state_msg.position = traj_left
            self.puppet_arm_left_publisher.publish(joint_state_msg)
            joint_state_msg.position = traj_right
            self.puppet_arm_right_publisher.publish(joint_state_msg)
            rate.sleep()

    def puppet_arm_publish_continuous_thread(self, left, right):
        if self.puppet_arm_publish_thread is not None:
            self.puppet_arm_publish_lock.release()
            self.puppet_arm_publish_thread.join()
            self.puppet_arm_publish_lock.acquire(False)
            self.puppet_arm_publish_thread = None
        self.puppet_arm_publish_thread = threading.Thread(target=self.puppet_arm_publish_continuous, args=(left, right))
        self.puppet_arm_publish_thread.start()

    def get_frame(self):
        if len(self.img_left_deque) == 0 or len(self.img_right_deque) == 0 or len(self.img_front_deque) == 0 or \
                (self.args.use_depth_image and (len(self.img_left_depth_deque) == 0 or len(self.img_right_depth_deque) == 0 or len(self.img_front_depth_deque) == 0)):
            return False
        if self.args.use_depth_image:
            frame_time = min([self.img_left_deque[-1].header.stamp.to_sec(), self.img_right_deque[-1].header.stamp.to_sec(), self.img_front_deque[-1].header.stamp.to_sec(),
                              self.img_left_depth_deque[-1].header.stamp.to_sec(), self.img_right_depth_deque[-1].header.stamp.to_sec(), self.img_front_depth_deque[-1].header.stamp.to_sec()])
        else:
            frame_time = min([self.img_left_deque[-1].header.stamp.to_sec(), self.img_right_deque[-1].header.stamp.to_sec(), self.img_front_deque[-1].header.stamp.to_sec()])

        if len(self.img_left_deque) == 0 or self.img_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.img_right_deque) == 0 or self.img_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.img_front_deque) == 0 or self.img_front_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.puppet_arm_left_deque) == 0 or self.puppet_arm_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.puppet_arm_right_deque) == 0 or self.puppet_arm_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if self.args.use_depth_image and (len(self.img_left_depth_deque) == 0 or self.img_left_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.args.use_depth_image and (len(self.img_right_depth_deque) == 0 or self.img_right_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.args.use_depth_image and (len(self.img_front_depth_deque) == 0 or self.img_front_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.args.use_robot_base and (len(self.robot_base_deque) == 0 or self.robot_base_deque[-1].header.stamp.to_sec() < frame_time):
            return False

        while self.img_left_deque[0].header.stamp.to_sec() < frame_time:
            self.img_left_deque.popleft()
        img_left = self.bridge.imgmsg_to_cv2(self.img_left_deque.popleft(), 'passthrough')

        while self.img_right_deque[0].header.stamp.to_sec() < frame_time:
            self.img_right_deque.popleft()
        img_right = self.bridge.imgmsg_to_cv2(self.img_right_deque.popleft(), 'passthrough')

        while self.img_front_deque[0].header.stamp.to_sec() < frame_time:
            self.img_front_deque.popleft()
        img_front = self.bridge.imgmsg_to_cv2(self.img_front_deque.popleft(), 'passthrough')

        while self.puppet_arm_left_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_left_deque.popleft()
        puppet_arm_left = self.puppet_arm_left_deque.popleft()

        while self.puppet_arm_right_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_right_deque.popleft()
        puppet_arm_right = self.puppet_arm_right_deque.popleft()

        img_left_depth = None
        if self.args.use_depth_image:
            while self.img_left_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_left_depth_deque.popleft()
            img_left_depth = self.bridge.imgmsg_to_cv2(self.img_left_depth_deque.popleft(), 'passthrough')

        img_right_depth = None
        if self.args.use_depth_image:
            while self.img_right_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_right_depth_deque.popleft()
            img_right_depth = self.bridge.imgmsg_to_cv2(self.img_right_depth_deque.popleft(), 'passthrough')

        img_front_depth = None
        if self.args.use_depth_image:
            while self.img_front_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_front_depth_deque.popleft()
            img_front_depth = self.bridge.imgmsg_to_cv2(self.img_front_depth_deque.popleft(), 'passthrough')

        robot_base = None
        if self.args.use_robot_base:
            while self.robot_base_deque[0].header.stamp.to_sec() < frame_time:
                self.robot_base_deque.popleft()
            robot_base = self.robot_base_deque.popleft()

        return (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
                puppet_arm_left, puppet_arm_right, robot_base)

    def img_left_callback(self, msg):
        if len(self.img_left_deque) >= 2000:
            self.img_left_deque.popleft()
        self.img_left_deque.append(msg)

    def img_right_callback(self, msg):
        if len(self.img_right_deque) >= 2000:
            self.img_right_deque.popleft()
        self.img_right_deque.append(msg)

    def img_front_callback(self, msg):
        if len(self.img_front_deque) >= 2000:
            self.img_front_deque.popleft()
        self.img_front_deque.append(msg)

    def img_left_depth_callback(self, msg):
        if len(self.img_left_depth_deque) >= 2000:
            self.img_left_depth_deque.popleft()
        self.img_left_depth_deque.append(msg)

    def img_right_depth_callback(self, msg):
        if len(self.img_right_depth_deque) >= 2000:
            self.img_right_depth_deque.popleft()
        self.img_right_depth_deque.append(msg)

    def img_front_depth_callback(self, msg):
        if len(self.img_front_depth_deque) >= 2000:
            self.img_front_depth_deque.popleft()
        self.img_front_depth_deque.append(msg)

    def puppet_arm_left_callback(self, msg):
        if len(self.puppet_arm_left_deque) >= 2000:
            self.puppet_arm_left_deque.popleft()
        self.puppet_arm_left_deque.append(msg)

    def puppet_arm_right_callback(self, msg):
        if len(self.puppet_arm_right_deque) >= 2000:
            self.puppet_arm_right_deque.popleft()
        self.puppet_arm_right_deque.append(msg)

    def robot_base_callback(self, msg):
        if len(self.robot_base_deque) >= 2000:
            self.robot_base_deque.popleft()
        self.robot_base_deque.append(msg)

    def ctrl_callback(self, msg):
        self.ctrl_state_lock.acquire()
        self.ctrl_state = msg.data
        self.ctrl_state_lock.release()

    def get_ctrl_state(self):
        self.ctrl_state_lock.acquire()
        state = self.ctrl_state
        self.ctrl_state_lock.release()
        return state

    def init_ros(self):
        rospy.init_node('joint_state_publisher', anonymous=True)
        rospy.Subscriber(self.args.img_left_topic, Image, self.img_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_right_topic, Image, self.img_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_front_topic, Image, self.img_front_callback, queue_size=1000, tcp_nodelay=True)
        if self.args.use_depth_image:
            rospy.Subscriber(self.args.img_left_depth_topic, Image, self.img_left_depth_callback, queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.args.img_right_depth_topic, Image, self.img_right_depth_callback, queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.args.img_front_depth_topic, Image, self.img_front_depth_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_arm_left_topic, JointState, self.puppet_arm_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_arm_right_topic, JointState, self.puppet_arm_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.robot_base_topic, Odometry, self.robot_base_callback, queue_size=1000, tcp_nodelay=True)
        self.puppet_arm_left_publisher = rospy.Publisher(self.args.puppet_arm_left_cmd_topic, JointState, queue_size=10)
        self.puppet_arm_right_publisher = rospy.Publisher(self.args.puppet_arm_right_cmd_topic, JointState, queue_size=10)
        self.robot_base_publisher = rospy.Publisher(self.args.robot_base_cmd_topic, Twist, queue_size=10)

# =================================================================================


class RealEnv2:
    """
    Environment for real robot bi-manual manipulation
    Action space:      [left_arm_qpos (6),             # absolute joint position
                        left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                        right_arm_qpos (6),            # absolute joint position
                        right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)

    Observation space: {"qpos": Concat[ left_arm_qpos (6),          # absolute joint position
                                        left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
                                        right_arm_qpos (6),         # absolute joint position
                                        right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
                        "qvel": Concat[ left_arm_qvel (6),         # absolute joint velocity (rad)
                                        left_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
                                        right_arm_qvel (6),         # absolute joint velocity (rad)
                                        right_gripper_qvel (1)]     # normalized gripper velocity (pos: opening, neg: closing)
                        "images": {"cam_high": (480x640x3),        # h, w, c, dtype='uint8'
                                   "cam_low": (480x640x3),         # h, w, c, dtype='uint8'
                                   "cam_left_wrist": (480x640x3),  # h, w, c, dtype='uint8'
                                   "cam_right_wrist": (480x640x3)} # h, w, c, dtype='uint8'
    """

    def __init__(self, init_node, setup_robots=True):
        self.ros_operator = RosOperator(ROSArgument())
        self.setup_robots()
        self.frame = None

    def setup_robots(self):
        left0 = [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156,
                 -0.00286102294921875, 0.00095367431640625, 3.557830810546875]
        right0 = [-0.00133514404296875, 0.00438690185546875, 0.034523963928222656, -0.053597450256347656,
                  -0.00476837158203125, -0.00209808349609375, 3.557830810546875]
        left1 = [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156,
                 -0.00286102294921875, 0.00095367431640625, -0.3393220901489258]
        right1 = [-0.00133514404296875, 0.00247955322265625, 0.01583099365234375, -0.032616615295410156,
                  -0.00286102294921875, 0.00095367431640625, -0.3397035598754883]

        self.ros_operator.puppet_arm_publish_continuous(left0, right0)
        self.ros_operator.puppet_arm_publish_continuous(left1, right1)

    def get_qpos(self):
        frame = self.frame
        (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
         puppet_arm_left, puppet_arm_right, robot_base) = frame

        # qpos（6关节 + 1夹爪）
        left_arm_qpos = np.array(puppet_arm_left.position[:6])
        left_gripper = np.array([puppet_arm_left.position[6]])
        right_arm_qpos = np.array(puppet_arm_right.position[:6])
        right_gripper = np.array([puppet_arm_right.position[6]])
        obs_qpos = np.concatenate([left_arm_qpos, left_gripper, right_arm_qpos, right_gripper])
        return obs_qpos

    def get_qvel(self):
        frame = self.frame
        (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
         puppet_arm_left, puppet_arm_right, robot_base) = frame

        # qvel（假设速度单位已经是rad/s）
        left_arm_qvel = np.array(puppet_arm_left.velocity[:6])
        left_gripper_vel = np.array([puppet_arm_left.velocity[6]])
        right_arm_qvel = np.array(puppet_arm_right.velocity[:6])
        right_gripper_vel = np.array([puppet_arm_right.velocity[6]])
        obs_qvel = np.concatenate([left_arm_qvel, left_gripper_vel, right_arm_qvel, right_gripper_vel])
        return obs_qvel

    def get_images(self):
        frame = self.frame
        (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
         puppet_arm_left, puppet_arm_right, robot_base) = frame
        image_dict = {
            "cam_high": cv2.resize(img_front, (640, 480)),
            "cam_left_wrist": cv2.resize(img_left, (640, 480)),
            "cam_right_wrist": cv2.resize(img_right, (640, 480)),
        }
        return image_dict

    # def _reset_joints(self):
    #     reset_position = START_ARM_POSE[:6]
    #     move_arms([self.puppet_bot_left, self.puppet_bot_right], [reset_position, reset_position], move_time=1)
    #
    # def _reset_gripper(self):
    #     """Set to position mode and do position resets: first open then close. Then change back to PWM mode"""
    #     move_grippers([self.puppet_bot_left, self.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)
    #     move_grippers([self.puppet_bot_left, self.puppet_bot_right], [PUPPET_GRIPPER_JOINT_CLOSE] * 2, move_time=1)

    def _get_obs(self):
        frame = self.ros_operator.get_frame()
        if frame:
            self.frame = frame
        else:
            rate = rospy.Rate(self.ros_operator.args.publish_rate)
            rate.sleep()
            # print("Frame did not update")
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos()
        obs['qvel'] = self.get_qvel()
        # obs['effort'] = self.get_effort()
        obs['images'] = self.get_images()

        return obs

    def get_observation(self, t=0):
        step_type = dm_env.StepType.FIRST if t == 0 else dm_env.StepType.MID
        return dm_env.TimeStep(
            step_type=step_type,
            reward=self.get_reward(),
            discount=None,
            observation=self._get_obs()
        )

    def get_reward(self):
        return 0

    def reset(self, fake=False):
        self.setup_robots()
        return self.get_observation()
        # if not fake:
        #     # Reboot puppet robot gripper motors
        #     self.puppet_bot_left.dxl.robot_reboot_motors("single", "gripper", True)
        #     self.puppet_bot_right.dxl.robot_reboot_motors("single", "gripper", True)
        #     self._reset_joints()
        #     self._reset_gripper()
        # return dm_env.TimeStep(
        #     step_type=dm_env.StepType.FIRST,
        #     reward=self.get_reward(),
        #     discount=None,
        #     observation=self._get_obs())

    def step(self, action):
        state_len = int(len(action) / 2)
        left_action = action[:state_len]
        right_action = action[state_len:]
        self.ros_operator.puppet_arm_publish(left_action, right_action)
        time.sleep(DT)
        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=self._get_obs())


# class RealEnv:
#     """
#     Environment for real robot bi-manual manipulation
#     Action space:      [left_arm_qpos (6),             # absolute joint position
#                         left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
#                         right_arm_qpos (6),            # absolute joint position
#                         right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)
#
#     Observation space: {"qpos": Concat[ left_arm_qpos (6),          # absolute joint position
#                                         left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
#                                         right_arm_qpos (6),         # absolute joint position
#                                         right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
#                         "qvel": Concat[ left_arm_qvel (6),         # absolute joint velocity (rad)
#                                         left_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
#                                         right_arm_qvel (6),         # absolute joint velocity (rad)
#                                         right_gripper_qvel (1)]     # normalized gripper velocity (pos: opening, neg: closing)
#                         "images": {"cam_high": (480x640x3),        # h, w, c, dtype='uint8'
#                                    "cam_low": (480x640x3),         # h, w, c, dtype='uint8'
#                                    "cam_left_wrist": (480x640x3),  # h, w, c, dtype='uint8'
#                                    "cam_right_wrist": (480x640x3)} # h, w, c, dtype='uint8'
#     """
#
#     def __init__(self, init_node, setup_robots=True):
#         self.puppet_bot_left = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper",
#                                                        robot_name=f'puppet_left', init_node=init_node)
#         self.puppet_bot_right = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper",
#                                                         robot_name=f'puppet_right', init_node=False)
#         if setup_robots:
#             self.setup_robots()
#
#         self.recorder_left = Recorder('left', init_node=False)
#         self.recorder_right = Recorder('right', init_node=False)
#         self.image_recorder = ImageRecorder(init_node=False)
#         self.gripper_command = JointSingleCommand(name="gripper")
#
#     def setup_robots(self):
#         setup_puppet_bot(self.puppet_bot_left)
#         setup_puppet_bot(self.puppet_bot_right)
#
#     def get_qpos(self):
#         left_qpos_raw = self.recorder_left.qpos
#         right_qpos_raw = self.recorder_right.qpos
#         left_arm_qpos = left_qpos_raw[:6]
#         right_arm_qpos = right_qpos_raw[:6]
#         left_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(left_qpos_raw[7])] # this is position not joint
#         right_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(right_qpos_raw[7])] # this is position not joint
#         return np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])
#
#     def get_qvel(self):
#         left_qvel_raw = self.recorder_left.qvel
#         right_qvel_raw = self.recorder_right.qvel
#         left_arm_qvel = left_qvel_raw[:6]
#         right_arm_qvel = right_qvel_raw[:6]
#         left_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(left_qvel_raw[7])]
#         right_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(right_qvel_raw[7])]
#         return np.concatenate([left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel])
#
#     def get_effort(self):
#         left_effort_raw = self.recorder_left.effort
#         right_effort_raw = self.recorder_right.effort
#         left_robot_effort = left_effort_raw[:7]
#         right_robot_effort = right_effort_raw[:7]
#         return np.concatenate([left_robot_effort, right_robot_effort])
#
#     def get_images(self):
#         return self.image_recorder.get_images()
#
#     def set_gripper_pose(self, left_gripper_desired_pos_normalized, right_gripper_desired_pos_normalized):
#         left_gripper_desired_joint = PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(left_gripper_desired_pos_normalized)
#         self.gripper_command.cmd = left_gripper_desired_joint
#         self.puppet_bot_left.gripper.core.pub_single.publish(self.gripper_command)
#
#         right_gripper_desired_joint = PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(right_gripper_desired_pos_normalized)
#         self.gripper_command.cmd = right_gripper_desired_joint
#         self.puppet_bot_right.gripper.core.pub_single.publish(self.gripper_command)
#
#     def _reset_joints(self):
#         reset_position = START_ARM_POSE[:6]
#         move_arms([self.puppet_bot_left, self.puppet_bot_right], [reset_position, reset_position], move_time=1)
#
#     def _reset_gripper(self):
#         """Set to position mode and do position resets: first open then close. Then change back to PWM mode"""
#         move_grippers([self.puppet_bot_left, self.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)
#         move_grippers([self.puppet_bot_left, self.puppet_bot_right], [PUPPET_GRIPPER_JOINT_CLOSE] * 2, move_time=1)
#
#     def _get_obs(self):
#         obs = collections.OrderedDict()
#         obs['qpos'] = self.get_qpos()
#         obs['qvel'] = self.get_qvel()
#         obs['effort'] = self.get_effort()
#         obs['images'] = self.get_images()
#         return obs
#
#     def get_observation(self, t=0):
#         step_type = dm_env.StepType.FIRST if t == 0 else dm_env.StepType.MID
#         return dm_env.TimeStep(
#             step_type=step_type,
#             reward=self.get_reward(),
#             discount=None,
#             observation=self._get_obs()
#         )
#
#     def get_reward(self):
#         return 0
#
#     def reset(self, fake=False):
#         if not fake:
#             # Reboot puppet robot gripper motors
#             self.puppet_bot_left.dxl.robot_reboot_motors("single", "gripper", True)
#             self.puppet_bot_right.dxl.robot_reboot_motors("single", "gripper", True)
#             self._reset_joints()
#             self._reset_gripper()
#         return dm_env.TimeStep(
#             step_type=dm_env.StepType.FIRST,
#             reward=self.get_reward(),
#             discount=None,
#             observation=self._get_obs())
#
#     def step(self, action):
#         state_len = int(len(action) / 2)
#         left_action = action[:state_len]
#         right_action = action[state_len:]
#         self.puppet_bot_left.arm.set_joint_positions(left_action[:6], blocking=False)
#         self.puppet_bot_right.arm.set_joint_positions(right_action[:6], blocking=False)
#         self.set_gripper_pose(left_action[-1], right_action[-1])
#         time.sleep(DT)
#         return dm_env.TimeStep(
#             step_type=dm_env.StepType.MID,
#             reward=self.get_reward(),
#             discount=None,
#             observation=self._get_obs())
#
#
# def get_action(master_bot_left, master_bot_right):
#     action = np.zeros(14) # 6 joint + 1 gripper, for two arms
#     # Arm actions
#     action[:6] = master_bot_left.dxl.joint_states.position[:6]
#     action[7:7+6] = master_bot_right.dxl.joint_states.position[:6]
#     # Gripper actions
#     action[6] = MASTER_GRIPPER_JOINT_NORMALIZE_FN(master_bot_left.dxl.joint_states.position[6])
#     action[7+6] = MASTER_GRIPPER_JOINT_NORMALIZE_FN(master_bot_right.dxl.joint_states.position[6])
#
#     return action


def make_real_env(init_node, setup_robots=True):
    env = RealEnv2(init_node, setup_robots)
    return env


# def test_real_teleop():
#     """
#     Test bimanual teleoperation and show image observations onscreen.
#     It first reads joint poses from both master arms.
#     Then use it as actions to step the environment.
#     The environment returns full observations including images.
#
#     An alternative approach is to have separate scripts for teleoperation and observation recording.
#     This script will result in higher fidelity (obs, action) pairs
#     """
#
#     onscreen_render = True
#     render_cam = 'cam_left_wrist'
#
#     # source of data
#     master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
#                                               robot_name=f'master_left', init_node=True)
#     master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
#                                                robot_name=f'master_right', init_node=False)
#     setup_master_bot(master_bot_left)
#     setup_master_bot(master_bot_right)
#
#     # setup the environment
#     env = make_real_env(init_node=False)
#     ts = env.reset(fake=True)
#     episode = [ts]
#     # setup visualization
#     if onscreen_render:
#         ax = plt.subplot()
#         plt_img = ax.imshow(ts.observation['images'][render_cam])
#         plt.ion()
#
#     for t in range(1000):
#         action = get_action(master_bot_left, master_bot_right)
#         ts = env.step(action)
#         episode.append(ts)
#
#         if onscreen_render:
#             plt_img.set_data(ts.observation['images'][render_cam])
#             plt.pause(DT)
#         else:
#             time.sleep(DT)
#
#
# if __name__ == '__main__':
#     test_real_teleop()