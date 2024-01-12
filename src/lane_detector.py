import time
import math
import numpy as np
import cv2
import rospy
import os
import pathlib

from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32
from std_msgs.msg import Float64MultiArray
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive


import argparse

parser = argparse.ArgumentParser()

# lane detector arguments
parser.add_argument('--output_dir', '-o', type=str, default='')
parser.add_argument('--output_freq', type=int, default=1,
                    help='output frame to folder. Requires output_dir not empty.')
parser.add_argument('--sat_cdf_lower_thres', type=float,
                    default=0.5, help='heuristic lower thres for saturation cdf.')
parser.add_argument('--blue_red_diff_thres', type=int,
                    default=30, help='blue and red channel difference thres')
parser.add_argument('--val_thres_percentile', type=int,
                    default=65, help='value percentile thres')
parser.add_argument('--red_val_tolerance', type=int, default=15)
parser.add_argument('--hue_thresh', type=str, default='15,40',
                    help='valid range for hue channel after HSL conversion')
parser.add_argument('--dilate_size', type=int, default=5,
                    help='kernel size for closing technique to remove noise')
parser.add_argument('--perspective_pts', '-p',
                    type=str, default='115,525,294,0', help='param for perpective projection. Four number represents respectively for src_leftx, src_rightx, laney, offsety.')
parser.add_argument('--window_height', type=int, default=20,
                    help='window height for lane fitting.')

# controller arguments
parser.add_argument('--steering_k', type=float, default=0.35)
parser.add_argument('--steering_i', type=float, default=1.8)
parser.add_argument('--curv_min', type=float, default=0.0)
parser.add_argument('--curv_max', type=float, default=0.4)
parser.add_argument('--angle_limit', type=float, default=30)
parser.add_argument('--vel_min', type=float, default=1.5)
parser.add_argument('--vel_max', type=float, default=1.5)
parser.add_argument('--look_ahead', type=float, default=-1.0,
                    help='fixed look ahead distance for pure pursuit controller. -1 denotes dynamic lookahead distance, which directly use the most distant waypoint as target point')
parser.add_argument('--max_look_ahead', type=float,
                    default=1.5, help='In case car looks too far away.')
parser.add_argument('--obstacle_tolerate_dist', type=float, default=-np.inf,
                    help='car change velocity if obstacle is within this distance. Negative infinity means the car does not avoid obstacle')

INCH2METER = 0.0254
PIX2METER_X = 0.0009525 # meter
PIX2METER_Y = 0.0018518 # meter
DIST_CAM2FOV_INCH = 21 # inch

class LaneDetector():
    def __init__(self, args, debug_mode=False):
        self.parse_params(args)
        self.debug_mode = debug_mode
        self.output_dir = args.output_dir
        self.output_freq = args.output_freq
        
        # controller params
        self.steering_k = args.steering_k
        self.steering_i = args.steering_i
        self.angle_limit = args.angle_limit
        self.curv_min = args.curv_min
        self.curv_max = args.curv_max
        self.vel_min = args.vel_min
        self.vel_max = args.vel_max
        self.look_ahead = args.look_ahead
        self.max_look_ahead = args.max_look_ahead
        self.obstacle_tolerate_dist = args.obstacle_tolerate_dist
        self.wheelbase = 0.325
        self.debug_mode = debug_mode
        
        self.way_pts = []
        self.reach_boundary = True
        self.cnt = 0
        # self.controller = F1tenth_controller(args)
        if not self.debug_mode:
            self.bridge = CvBridge()
            self.sub_image = rospy.Subscriber('/D435I/color/image_raw', Image, self.img_callback, queue_size=1)

            self.ctrl_pub = rospy.Publisher("/vesc/low_level/ackermann_cmd_mux/input/navigation", AckermannDriveStamped, queue_size=1)
            self.drive_msg = AckermannDriveStamped()
            self.drive_msg.header.frame_id = "f1tenth_control"


    def parse_params(self, args):
        # parse params
        # self.sat_thres_min, self.sat_thres_max = args.sat_thresh.split(',')
        # self.sat_thres_min, self.sat_thres_max = int(self.sat_thres_min), int(self.sat_thres_max)
        # assert self.sat_thres_min < self.sat_thres_max

        self.hue_thres_min, self.hue_thres_max = args.hue_thresh.split(',')
        self.hue_thres_min, self.hue_thres_max = int(self.hue_thres_min), int(self.hue_thres_max)
        assert self.hue_thres_min < self.hue_thres_max

        src_leftx, src_rightx, laney, offsety = args.perspective_pts.split(',')
        self.src_leftx, self.src_rightx, self.laney, self.offsety = int(
            src_leftx), int(src_rightx), int(laney), int(offsety)
        
        self.val_thres_percentile = args.val_thres_percentile
        self.dilate_size = args.dilate_size
        self.sat_cdf_lower_thres = args.sat_cdf_lower_thres
        self.blue_red_diff_thres = args.blue_red_diff_thres
        self.red_val_tolerance = args.red_val_tolerance
        self.window_height = args.window_height
        
    def img_callback(self, data):

        try:
            # Convert a ROS image message into an OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        start_time = time.time()
        
        raw_img = cv_image.copy()
        ret = self.detection(raw_img)
        print("Detection takes time: {:.3f} seconds".format(time.time() - start_time))

        
        latest_way_pts, latest_reach_boundary = self.get_latest_info()
        self.run(latest_way_pts, latest_reach_boundary)
                
        # output images for debug
        self.cnt += 1
        OUTPUT_DIR = os.path.join('test_images', self.output_dir)
        pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        if self.output_dir != '' and self.cnt % self.output_freq == 0:
            output_path = '{}/{}.png'.format(OUTPUT_DIR, self.cnt)
            print ('output to {}'.format(output_path))
            cv2.imwrite(output_path, raw_img)
         
        # _, _, way_pts = ret
        # min_way_pts = 3
        # if len(way_pts) < min_way_pts:
        #     print ('Number of detected way_pts is less than {}. Skip this frame.'.format(min_way_pts))
        #     return
        # else:
        #     self.controller.run(way_pts)
        
    def line_fit(self, binary_warped):
        """
        Find and fit lane lines
        """
        ### 1. sliding window to find the base point
        height, width = binary_warped.shape
        # nwindows = 15
        sliding_offset = 5
        margin = 70
        best_base_x = -1
        best_num_pixels = 0
        
        for basex in range(margin, width-margin, sliding_offset):
            left = basex - margin
            right = basex + margin
            total_num_pixels = cv2.countNonZero(
                binary_warped[-self.window_height:, left:right])
            
            if total_num_pixels > best_num_pixels:
                best_num_pixels = total_num_pixels
                best_base_x = basex
        
        if best_base_x == -1:
            return None
        
        # visualize
        # vis = cv2.cvtColor(binary_warped*255, cv2.COLOR_GRAY2BGR)
        # vis = cv2.rectangle(
        #     vis, (best_base_x - margin, height - self.window_height), 
        #     (best_base_x + margin, height), 
        #     (0, 0, 255))
        # imshow("vis", vis)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
    
        # Set minimum number of pixels found to recenter window
        minpix = 200
        
        # Step through the windows one by one
        color_warped = cv2.cvtColor(binary_warped*255, cv2.COLOR_GRAY2BGR)
        color_warped[color_warped > 0] = 255
            
        basex = best_base_x
        lane_pts = []
        prev_basex_list = []

        i = 0
        reach_boundary = False
        while True:
            win_top = height - (i + 1) * self.window_height
            win_bottom = win_top + self.window_height
            
            # adjust basex based on the slope of previous two basex
            if i >= 2:
                slope = prev_basex_list[-1] - prev_basex_list[-2]
                basex = prev_basex_list[-1] + slope
                
            window_inds = np.where((nonzerox > basex - margin) &
                                (nonzerox < basex + margin) &
                                (nonzeroy > win_top) &
                                (nonzeroy < win_bottom))
            window_nonzerox = nonzerox[window_inds]
            window_nonzeroy = nonzeroy[window_inds]
            # print (len(window_nonzerox))
            
            # finish fitting condition
            reach_boundary = basex - margin < 0 or \
                            basex + margin >= width or \
                            win_top < 0

            if len(window_nonzerox) < minpix:
                break
            
            # correct basex by average and use average (x, y) as way points
            basex = int(np.mean(window_nonzerox))
            basey = int(np.mean(window_nonzeroy))
            lane_pts.append([basex, basey])
            prev_basex_list.append(basex)
                
            i += 1
            
            # visualization
            color_warped = cv2.rectangle(
                color_warped, (basex - margin, win_top), (basex +margin, win_bottom), (0, 0, 255))
            # imshow("color_warped", color_warped)
        
        lanex = [pt[0] for pt in lane_pts]
        laney = [pt[1] for pt in lane_pts]
        # vis lane points
        for x, y in zip(lanex, laney):
            color_warped = cv2.circle(color_warped, (x, y), 2, (0,255, 0), -1)
        
        # try:
        #     lane_fit = np.polyfit(laney, lanex, deg=2)
                    
        #     ### vis points nonzero ###
        #     # for x, y in zip(rightx, righty):
        #     #     color_warped = cv2.circle(color_warped, (x, y), 1, (0,255, 0), -1)
        #     # imshow("points", color_warped )

        # except TypeError:
        #     print("Unable to detect lanes")
        #     return None

        ret = {}
        ret['vis_warped'] = color_warped
        ret['lanex'] = lanex
        ret['laney'] = laney
        ret['reach_boundary'] = reach_boundary
        return ret
    
    def color_thresh(self, img):
        """
        Convert RGB to HSL and threshold to binary image using S channel
        """
        # 1. Convert the image from RGB to HSL
        # 2. Apply threshold on S channel to get binary image
        # Step 1: Filter out pixels with strong reflection
        img = img.copy()
        blue_channel = img[:, :, 0].astype(np.float32)
        red_channel = img[:, :, 2].astype(np.float32)
        blud_red_diff_cond = red_channel - blue_channel > self.blue_red_diff_thres

        hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        
        # For HSL
        # ref: https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html#color_convert_rgb_hls
        #   image format: (8-bit images) V ← 255⋅V, S ← 255⋅S, H ← H/2(to fit to 0 to 255)

        # Step 2: Apply threshold on the S (Saturation) channel to get a binary image
        h, l, s = cv2.split(hls_img)
        binary_output = np.zeros_like(l)
        
        # dynamic search sat_thres_min
        s_warped, M, Minv = self.perspective_transform(s)
        sat_hist, bins = np.histogram(s_warped.flatten(), bins=256, range=[0, 256])

        # Calculate the cumulative distribution function (CDF) of the histogram
        cdf = sat_hist.cumsum()
        cdf_normalized = cdf / cdf.max() # Normalize the CDF to the range [0, 1]
        bin_idxs = \
            np.where((cdf_normalized > self.sat_cdf_lower_thres) & (cdf_normalized < 0.90))[0]
        sat_thres_min = np.argmin( [sat_hist[idx] for idx in bin_idxs] ) + bin_idxs[0]
        sat_cond = ((sat_thres_min <= s) & (s <= 255))
        
        
        # Steps 2: Apply value threshold on image
        # Use red channel of raw_image instead of l channel to do the value filtering
        # Because red channel for yellow lane is much different from background
        red_channel = img[:, :, 2] # red channel
        red_channel_warped, M, Minv = self.perspective_transform(red_channel)
        val_thres_min = np.percentile(red_channel_warped, self.val_thres_percentile)
        # val_mean = np.mean(red_channel_warped)
        val_cond = (val_thres_min <= red_channel) & (red_channel <= 255)

        mean_red_val = np.mean(red_channel_warped)
        mean_red_val_cond = mean_red_val + self.red_val_tolerance <= red_channel    
    
        # Step 3: Apply predefined hue threshold on image
        hue_cond = (self.hue_thres_min <= h) & (h <= self.hue_thres_max)
        
        # combine conditions and get final output
        binary_output[val_cond & sat_cond & hue_cond & blud_red_diff_cond & mean_red_val_cond] = 1

        # closing
        kernel = np.ones((self.dilate_size, self.dilate_size), np.uint8)
        binary_output = cv2.morphologyEx(
            binary_output.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

        return binary_output

    def perspective_transform(self, img, verbose=False):
        """
        Get bird's eye view from input image
        """
        # 1. Visually determine 4 source points and 4 destination points
        # 2. Get M, the transform matrix, and Minv, the inverse using cv2.getPerspectiveTransform()
        # 3. Generate warped image in bird view using cv2.warpPerspective()

        # Define four points as (x, y) coordinates
        src_height, src_width = img.shape[:2]

        src_pts = np.array([[self.src_leftx, self.laney],
                            [0, src_height - self.offsety],
                            [src_width, src_height - self.offsety],
                            [self.src_rightx, self.laney]], dtype=np.int32)

        # dst_width, dst_height = 720, 1250
        dst_width, dst_height = src_width, src_height
        dst_pts = np.array([[0, 0],
                            [0, dst_height],
                            [dst_width, dst_height],
                            [dst_width, 0]], dtype=np.int32)

        def calc_warp_points():
            src = np.float32(src_pts)
            dst = np.float32(dst_pts)

            return src, dst

        src, dst = calc_warp_points()
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)

        # keep same size as input image
        warped_img = cv2.warpPerspective(
            img, M, (dst_width, dst_height), flags=cv2.INTER_NEAREST)

        return warped_img, M, Minv

    def get_latest_info(self):
        return self.way_pts, self.reach_boundary
    
    def update_waypoints(self, lanex, laney, width, height, reach_boundary):
            
            # transform from image coord (x, y) to camera coord in meters
            lanex = [(x - width // 2) * PIX2METER_X for x in lanex]
            laney = [(height - y) * PIX2METER_Y + DIST_CAM2FOV_INCH * INCH2METER for y in laney]
            
            # print ('\n--------- waypoints ---------')
            # for i, (x, y) in enumerate(zip(lanex, laney)):
            #     print ('{} => Jacky coord: ({:.2f}, {:.2f}), Ricky coord: ({:.2f}, {:.2f})'
            #            .format(i+1, x, y, y, -x))
            
            # change to Ricky's coordinate    
            way_pts = [(y, -x) for x, y in zip(lanex, laney)]
            
            # only update way pts when succefully fit lines
            if len(way_pts) >= 3:
                self.way_pts = way_pts
                self.reach_boundary = reach_boundary
            else:
                print ('Number of detected way_pts < 3. Use waypoints of previous frame')
    
    def get_matrix_calibration(self, img_shape,
                               src_leftx=218,
                               src_rightx=467,
                               laney=348,
                               offsety=0):
        """
        Get bird's eye view from input image
        """
        # 1. Visually determine 4 source points and 4 destination points
        # 2. Get M, the transform matrix, and Minv, the inverse using cv2.getPerspectiveTransform()
        # 3. Generate warped image in bird view using cv2.warpPerspective()

        # Define four points as (x, y) coordinates
        src_height, src_width = img_shape

        src_pts = np.array([[src_leftx, laney],
                            [0, src_height - offsety],
                            [src_width, src_height - offsety],
                            [src_rightx, laney]], dtype=np.int32)

        # dst_width, dst_height = 720, 1250
        dst_width, dst_height = src_width, src_height
        dst_pts = np.array([[0, 0],
                            [0, dst_height],
                            [dst_width, dst_height],
                            [dst_width, 0]], dtype=np.int32)

        def calc_warp_points():
            src = np.float32(src_pts)
            dst = np.float32(dst_pts)
            return src, dst

        src, dst = calc_warp_points()
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)

        return M, Minv

    def convert2CalibrationCoord(self, shape, lanex, laney, Minv):
        clb_M, clb_Minv = self.get_matrix_calibration(shape)
        num_waypts = len(lanex)
        coords = np.zeros((num_waypts, 2))
        for i, (x, y) in enumerate(zip(lanex, laney)):
            coords[i][0] = x
            coords[i][1] = y

        one_vec = np.ones((num_waypts, 1)) # convert to homogeneous coord
        coords = np.concatenate((coords, one_vec), axis=1)
        
        raw_coords = Minv @ coords.T
        clb_coords = clb_M @ raw_coords
        clb_coords_normalize = clb_coords / clb_coords[2]
        clb_coords_normalize = clb_coords_normalize.T
        
        # display coord transformation
        # for i, (x, y) in enumerate(zip(lanex, laney)):
        #     print (f'{i+1} => {(x, y)} => {clb_coords_normalize[i][:2]}')
            
        return clb_coords_normalize[:, 0], clb_coords_normalize[:, 1]
            
    def detection(self, img):
        # use color_thresh result only
        color_output = self.color_thresh(img)
        color_warped, M, Minv = self.perspective_transform(color_output)
        
        # line fit
        ret = self.line_fit(color_warped)
        if ret is None: # fail to polyfit waypoints
            print ('Cannot find a single waypoint. Use waypoints of previous frame')
            return None
            
        # convert to calibration coords
        clb_x, clb_y = \
            self.convert2CalibrationCoord(img.shape[:2], ret['lanex'], ret['laney'], Minv)
        
        # get get_waypoints
        height, width = img.shape[:2]
        self.update_waypoints(clb_x, clb_y, width, height, ret['reach_boundary'])
        
        return ret['reach_boundary'], ret['vis_warped'], cv2.cvtColor(color_warped, cv2.COLOR_GRAY2BGR), self.way_pts

    def get_steering_based_point(self, targ_pts, look_ahead=None):
        """ 
            Extend the curve to find the most suitable way point
        """
        lanex = [pt[0] for pt in targ_pts]
        laney = [pt[1] for pt in targ_pts]
        
        if look_ahead is None:
            look_ahead = self.look_ahead

        # num_waypt = len(lanex)
        # if num_waypt > 10:
        #     look_ahead = 1.5
            
        lane_fit = np.polyfit(lanex, laney, deg=2)
        steering_based_pt = [-1, -1]
        x = lanex[0]
        while True:
            y = np.polyval(lane_fit, x)
            dist = np.hypot(x, y)
            steering_based_pt = [x, y]
            if dist > look_ahead:
                break
            
            x += 0.01 # meter
            
        return steering_based_pt
        
        
    def run(self, way_pts=None, reach_boundary=True):
        ## find the goal point which is the last in the set of points less than lookahead distance
        if way_pts is None: # way_pts is provided by the perload file
            self.get_targ_points()
        else:
            self.targ_pts = way_pts
            
        ### Determine speed ###
        target_velocity = self.vel_max # constant speed
        obs_detected = False # obstacle detected
        
        ### lateral control using pure pursuit ###
        
        ## compute track curvature for longititudal control
        num_waypts = len(self.targ_pts)
        idxs = [0, num_waypts // 2, num_waypts - 1]
        if len(self.targ_pts) >= 3:
            dx0 = self.targ_pts[idxs[1]][0] - self.targ_pts[idxs[0]][0]
            dy0 = self.targ_pts[idxs[1]][1] - self.targ_pts[idxs[0]][1]
            dx1 = self.targ_pts[idxs[2]][0] - self.targ_pts[idxs[1]][0]
            dy1 = self.targ_pts[idxs[2]][1] - self.targ_pts[idxs[1]][1]
    
            # dx0 = self.targ_pts[-2][0] - self.targ_pts[-3][0]
            # dy0 = self.targ_pts[-2][1] - self.targ_pts[-3][1]
            # dx1 = self.targ_pts[-1][0] - self.targ_pts[-2][0]
            # dy1 = self.targ_pts[-1][1] - self.targ_pts[-2][1]
            ddx, ddy = dx1 - dx0, dy1 - dy0
            curvature = np.inf if dx1 == 0 and dy1 == 0 else abs((dx1*ddy - dy1*ddx) / (dx1**2 + dy1**2) ** (3/2))
        else:
            curvature = np.inf

        ## adjust speed according to curvature and steering angle
        curvature = min(self.curv_max, curvature)
        curvature = max(self.curv_min, curvature)
        target_velocity = self.vel_max - (self.vel_max - self.vel_min) * curvature / (self.curv_max - self.curv_min)
        
        # Determine the target(self.goal_x, self.goal_y) point
        if self.look_ahead > 0:
            print ("fixed look_ahead distance")
            self.goal_x, self.goal_y = self.get_steering_based_point(self.targ_pts)
        else:
            print ("dynamic look_ahead distance")
            # Use the last way point as our target point
            self.goal_x = self.targ_pts[-1][0]
            self.goal_y = self.targ_pts[-1][1]
            
            # look-ahead distance between the farthest waypoint and current position
            ld_farthest_waypt = np.hypot(self.goal_x, self.goal_y)
            
            # Check if obstacle is right in the front.
            if ld_farthest_waypt < self.obstacle_tolerate_dist and not reach_boundary:
                obs_detected = True
                target_velocity = 0
            
            # Find again the waypoint if look-ahead distance is too large.
            # Car does not need to see that far.
            if ld_farthest_waypt > self.max_look_ahead:
                self.goal_x, self.goal_y = self.get_steering_based_point(self.targ_pts, self.max_look_ahead)
            
        # final look-ahead distance between a waypoint and current position
        ld = np.hypot(self.goal_x, self.goal_y)
            
        ### Determine target steering angle (tuning this part as needed) ###
        alpha = np.arctan2(self.goal_y, self.goal_x)
        angle = np.arctan2((self.steering_k * 2 * self.wheelbase * np.sin(alpha)) / ld, 1) * self.steering_i
        target_steering = round(np.clip(angle, -np.radians(self.angle_limit), np.radians(self.angle_limit)), 3)
        target_steering_deg = round(np.degrees(target_steering)) # for msg display only
        
        # if target_steering >= np.radians(20):
        #     target_velocity = self.vel_min
            
            
            
        if not self.debug_mode:
            self.drive_msg.header.stamp = rospy.get_rostime()
            self.drive_msg.drive.steering_angle = target_steering
            self.drive_msg.drive.speed = target_velocity
            self.ctrl_pub.publish(self.drive_msg)
        
        msgs = [
            "max lookahead: {:.2f} meters".format(ld),
            # "2. last waypt dist: {:.2f}".format(ld_farthest_waypt),
            "last waypt: ({:.2f}, {:.2f})".format(self.targ_pts[-1][0], self.targ_pts[-1][1]),
            "target_pt: ({:.2f}, {:.2f})".format(self.goal_x, self.goal_y),
            "steering(deg): {} degree".format(target_steering_deg),
            "target_vel: {:.2f}".format(target_velocity),
            "curvature: {:.2f}".format(curvature),
            # "reach_boundary: {}".format(reach_boundary),
            "obs_detected: {}".format(obs_detected),
        ]
        # if self.look_ahead <= 0:
        #     msgs = ["last waypt dist: {:.2f}".format(ld_farthest_waypt)] + msgs
        
        # print msgs
        print ('\n----- control msgs -----')
        for i, msg in enumerate(msgs):
            print ('{}. {}'.format(i+1, msg))

        return msgs # return msgs for debug
    
def main():
    args = parser.parse_args()
    print ('======= Initial arguments =======')
    params = []
    for key, val in vars(args).items():
        param = f"--{key} {val}"
        print(f"{key} => {val}")
        params.append(param)

    # save params for debug
    if args.output_dir != '':
        OUTPUT_DIR = os.path.join('test_images', args.output_dir)
        pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)  
        with open(os.path.join(OUTPUT_DIR, 'params.txt'), 'w') as f:
            commands = ['python debug_helper.py -i {}'.format(OUTPUT_DIR)] + params
            f.write(' \\\n  '.join(commands))

    assert args.curv_min <= args.curv_max
    assert args.vel_min <= args.vel_max
    
    rospy.init_node('rgb_track_node', anonymous=True)
    rate = rospy.Rate(15)  # Hz

    lane_detector = LaneDetector(args)
    # controller = F1tenth_controller(args)
    try:
        print ('\nStart navigation...')
        while not rospy.is_shutdown():
            # start_time = time.time()
            # way_pts = lane_detector.get_latest_info()
            
            # Do not update control signal. 
            # Because it cannot fit polyline if way points < 3
            # if len(way_pts) >= 3:
            #     controller.run(way_pts)
            rate.sleep()  # Wait a while before trying to get a new waypoints
            # print("pipeline takes time: {:.3f} seconds".format(time.time() - start_time))
            # pass
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()