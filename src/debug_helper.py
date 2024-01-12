#!/usr/bin/env python3

from __future__ import print_function

# ROS Headers
import rospy
from lane_detector import LaneDetector
import argparse
import pathlib
import os
import cv2
import numpy as np
from lane_detector import parser

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# debug params
parser.add_argument('--specified_name', '-s', type=str, help='Specify image name to debug.')
parser.add_argument('--num_samples', '-n', type=int, default=-1,
                    help="-1 means check all files in the input folder")
parser.add_argument('--input_dir', '-i', type=str, default='test_images')
parser.add_argument("--use_rosbag", action="store_true") 

args = parser.parse_args()

OUTPUT_DIR = 'debug_results/debug_results_new'

# ctrl_params displayed on results
ctrl_params = [
    'obs_tolerate: {}'.format(args.obstacle_tolerate_dist),
    'max_look_ahead: {}'.format(args.max_look_ahead),
    'look_ahead: {}'.format(args.look_ahead),
    'steering_k: {}'.format(args.steering_k),
    'steering_i: {}'.format(args.steering_i),
    'angle_limit: {}'.format(args.angle_limit),
    'vel_min: {}'.format(args.vel_min),
    'vel_max: {}'.format(args.vel_max),
]


def get_output_img(raw_img, vis_warped, ctrl_msgs, way_pts):
    height, width = raw_img.shape[:2]
    canvas = np.zeros((height//3, width, 3), dtype=raw_img.dtype) + 50
    concat = cv2.vconcat([raw_img, canvas, vis_warped])

    # puttext params
    font_scale = 0.6
    font_color = (0, 0, 255)
    font_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    padding = 10
    (box_x, box_y), _ = cv2.getTextSize(
            "dummy", font, font_scale, font_thickness)

    # top left
    pos = [padding, height + 30]
    cv2.putText(concat, "Params:", pos, font,
                font_scale, font_color, font_thickness)
    for msg in ctrl_params:
        pos[1] = pos[1] + box_y + padding
        cv2.putText(concat, msg, pos, font,
                    font_scale, font_color, font_thickness)
        (box_x, box_y), _ = cv2.getTextSize(
            msg, font, font_scale, font_thickness)

    # top center
    font_color = (0, 128, 255)
    pos = [padding + width//3, height + 30]
    cv2.putText(concat, "Control signal:", pos, font,
                font_scale, font_color, font_thickness)
    
    for msg in ctrl_msgs:
        pos[1] = pos[1] + box_y + padding
        cv2.putText(concat, msg, pos, font,
                    font_scale, font_color, font_thickness)
        
    # top right
    font_color = (0, 255, 0)
    pos = [width - 150, height + 30]
    cv2.putText(concat, "Way pts:", pos, font,
                font_scale, font_color, font_thickness)
    for way_pt in way_pts:
        pos[1] = pos[1] + box_y + padding
        text = '({:.2f}, {:.2f})'.format(way_pt[0], way_pt[1])
        cv2.putText(concat, text, pos, font,
                    font_scale, font_color, font_thickness)
        
    return concat

def run_on_folder(img_path, lane_detector, fail_paths):
        img_name = img_path.split('/')[-1]
        raw_img = cv2.imread(img_path)
        ret_lane = lane_detector.detection(raw_img)
        if ret_lane is None:
            fail_paths.append(img_path)
            return
        reach_boundary, vis_warped, color_warped, way_pts = ret_lane
        controller = lane_detector
        ctrl_msgs = controller.run(way_pts, reach_boundary)

        out_img = get_output_img(raw_img, vis_warped, ctrl_msgs, way_pts)
        output_path = os.path.join(OUTPUT_DIR, img_name)
        print ('output_path:', output_path)
        cv2.imwrite(output_path, out_img)

class Debugger():
    def __init__(self, lane_detector):
        self.lane_detector = lane_detector
        # self.controller = controller
        self.bridge = CvBridge()
        self.sub_image = rospy.Subscriber('/D435I/color/image_raw', Image, self.img_callback, queue_size=1)
        self.pub_image = rospy.Publisher(
            "lane_detection/annotate", Image, queue_size=1)
        
    def img_callback(self, data):
        try:
            # Convert a ROS image message into an OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        
        raw_img = cv_image.copy()
        ret_lane = self.lane_detector.detection(raw_img)
        if ret_lane is None:
            return
        reach_boundary, vis_warped, color_warped, way_pts = ret_lane
        
        # ctrl_msgs = self.controller.run(way_pts)
        ctrl_msgs = self.lane_detector.run(way_pts)

        out_img = get_output_img(raw_img, vis_warped, ctrl_msgs, way_pts)
        out_img_msg = self.bridge.cv2_to_imgmsg(out_img, 'bgr8')

        # Publish image message in ROS
        self.pub_image.publish(out_img_msg)

def main():
    # args = parser.parse_args()
    print('======= Initial arguments =======')
    for key, val in vars(args).items():
        print(f"{key} => {val}")

    assert args.curv_min <= args.curv_max
    assert args.vel_min <= args.vel_max

    lane_detector = LaneDetector(args, debug_mode=True)
    # controller = F1tenth_controller(args, debug_mode=True)

    pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    if args.use_rosbag:
        rospy.init_node('rgb_track_node', anonymous=True)
        rate = rospy.Rate(30)  # Hz    
        print ('\nStart navigation...')
        # Debugger(lane_detector, controller)
        Debugger(lane_detector)
        while not rospy.is_shutdown():
            rate.sleep()  # Wait a while before trying to get a new waypoints
    else:
        fail_paths = []
        if args.specified_name:
            img_path = os.path.join(
                args.input_dir, '{}.png'.format(args.specified_name))
            print ('img_path:', img_path)
            run_on_folder(img_path, lane_detector, fail_paths)
        else:
            paths = sorted(os.listdir(args.input_dir))
            for i, img_path in enumerate(paths):
                if i == args.num_samples:
                    break
                if not img_path.endswith('png') and not img_path.endswith('jpg'):
                    continue
                
                img_path = os.path.join(args.input_dir, img_path)
                print ('img_path:', img_path)
                run_on_folder(img_path, lane_detector, fail_paths)
        
        print ("\n ----- {} failed images -----".format(len(fail_paths)))
        for i, path in enumerate(fail_paths):
            print ('{}. {}'.format(i+1, path))
            
if __name__ == '__main__':
    main()
