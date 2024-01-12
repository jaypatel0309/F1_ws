import time
import math
import numpy as np
import cv2
from skimage import morphology
import argparse
import matplotlib.pyplot as plt
import pathlib
import os

parser = argparse.ArgumentParser()
parser.add_argument('--vis_mode', '-v', action='store_true',
                    help='turn on to see visualization image by imshow for each step')
parser.add_argument('--vis_output', '--vo', action='store_true',
                    help='output visualization results')
parser.add_argument('--num_samples', '-n', type=int, default=-1,
                    help='num of images to check. -1 means check all images under the input folder')
parser.add_argument('--vis_hue', action='store_true',
                    help='turn on to display results filtered by different hue range. Results can be found at ./hue-test')
parser.add_argument('--vis_sat', action='store_true',
                    help='turn on to display results filtered by different saturation. Results can be found at ./sat-test')
parser.add_argument('--append_str', '-a', type=str,
                    default='tmp', help='appended string to output directory')
parser.add_argument('--specified_name', '-s', type=str,
                    help='Specify image name to debug.')
parser.add_argument('--gradient_thresh', '-g', type=str, default='75,150')
# parser.add_argument('--sat_thresh', type=str, default='60,255')
parser.add_argument('--sat_cdf_lower_thres', type=float, default=0.5,
                    help='search cdf bin that above this threshold. Define the saturation bin with minimum num pixels is the boundary of background/foreground.')
# parser.add_argument('--val_thresh', type=str, default='80,255')
# parser.add_argument('--val_thres_offset', type=int, default=20)
parser.add_argument('--val_thres_percentile', type=int, default=65,
                    help='pixel values that are below this percentil will be assigned zero. This helps filtering by value')
parser.add_argument('--val_reflection_thres', type=int, default=210,
                    help='pixel values in blue channel that are above this thres will be assigned zero. This avoids strong reflection affects filtering result')
parser.add_argument('--hue_thresh', type=str,
                    default='15,40', help='for hue filtering')
parser.add_argument('--dilate_size', type=int, default=5,
                    help='Closing small holes inside the yellow lane')
parser.add_argument('--window_height', type=int, default=20)
parser.add_argument('--perspective_pts', '-p',
                    type=str, default='218,467,348,0')
parser.add_argument('--input_dir', '-i', type=str,
                    default='/Users/jackyyeh/Desktop/Courses/UIUC/ECE484-Principles-Of-Safe-Autonomy/ECE484-Principles-Of-Safe-Autonomy-2023Fall-Final/f1tenth_ros1_ws/src/frames')

args = parser.parse_args()

TMP_DIR = './vis_{}'.format(args.append_str)
grad_thres_min, grad_thres_max = args.gradient_thresh.split(',')
grad_thres_min, grad_thres_max = int(grad_thres_min), int(grad_thres_max)
assert grad_thres_min < grad_thres_max

# val_thres_min, val_thres_max = args.val_thresh.split(',')
# val_thres_min, val_thres_max = int(val_thres_min), int(val_thres_max)
# assert val_thres_min < val_thres_max

# sat_thres_min, sat_thres_max = args.sat_thresh.split(',')
# sat_thres_min, sat_thres_max = int(sat_thres_min), int(sat_thres_max)

hue_thres_min, hue_thres_max = args.hue_thresh.split(',')
hue_thres_min, hue_thres_max = int(hue_thres_min), int(hue_thres_max)
assert hue_thres_min < hue_thres_max

src_leftx, src_rightx, laney, offsety = args.perspective_pts.split(',')
src_leftx, src_rightx, laney, offsety = int(
    src_leftx), int(src_rightx), int(laney), int(offsety)

img_name = ""

INCH2METER = 0.0254
PIX2METER_X = 0.0009525  # meter
PIX2METER_Y = 0.0018518  # meter
DIST_CAM2FOV_INCH = 21  # inch


def putText(img, text,
            font_scale=1,
            font_color=(0, 0, 255),
            font_thickness=2,
            font=cv2.FONT_HERSHEY_SIMPLEX):

    # Calculate the position for bottom right corner
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = img.shape[1] - text_size[0] - 10  # Adjust for padding
    text_y = img.shape[0] - 10  # Adjust for padding

    cv2.putText(img, text, (text_x, text_y), font,
                font_scale, font_color, font_thickness)


def imshow(window_name, image):
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def gradient_thresh(img, thresh_min=grad_thres_min, thresh_max=grad_thres_max):
    """
    Apply sobel edge detection on input image in x, y direction
    """
    # 1. Convert the image to gray scale
    # 2. Gaussian blur the image
    # 3. Use cv2.Sobel() to find derievatives for both X and Y Axis
    # 4. Use cv2.addWeighted() to combine the results
    # 5. Convert each pixel to uint8, then apply threshold to get binary image

    # Step 1: Load the image and convert it to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 2: Apply Gaussian blur to the grayscale image
    blurred_image = cv2.GaussianBlur(gray_img, (5, 5), 0)

    # Step 3: Use cv2.Sobel() to find derivatives for both X and Y axes
    sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)

    # Step 4: Combine the results using cv2.addWeighted()
    sobel_combined = cv2.addWeighted(np.absolute(
        sobel_x), 0.5, np.absolute(sobel_y), 0.5, 0)

    # Step 5: Convert each pixel to uint8 and apply a threshold to get a binary image
    sobel_combined = np.uint8(sobel_combined)
    binary_output = np.zeros_like(sobel_combined)
    binary_output[(thresh_min < sobel_combined) &
                  (sobel_combined < thresh_max)] = 1

    # closing
    # kernel = np.ones((5, 5), np.uint8)
    # binary_output = cv2.morphologyEx(
    #     binary_output.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    # vis
    # if args.vis_mode:
    #     vis = cv2.cvtColor(binary_output*255, cv2.COLOR_GRAY2BGR)
    #     imshow("binary_output", cv2.hconcat([img, vis]))

    return binary_output


def vis_hls_hist(h, l, s):
    # tmp to vis saturaion
    h_warped, M, Minv = perspective_transform(h)
    l_warped, M, Minv = perspective_transform(l)
    s_warped, M, Minv = perspective_transform(s)

    # Calculate the histogram of the saturation channel
    histogram, bins = np.histogram(
        s_warped.flatten(), bins=256, range=[0, 256])

    # Calculate the cumulative distribution function (CDF) of the histogram
    cdf = histogram.cumsum()

    # Normalize the CDF to the range [0, 1]
    cdf_normalized = cdf * histogram.max() / cdf.max()

    # Plot histograms
    plt.figure(figsize=(15, 5))

    # Plot H channel histogram
    plt.subplot(131)
    plt.hist(h_warped.flatten(), bins=256, color='r', range=[0, 256])
    plt.title('H Channel Histogram')

    # Plot L channel histogram
    plt.subplot(132)
    plt.hist(l_warped.flatten(), bins=256, color='g', range=[0, 256])
    plt.title('L Channel Histogram')

    # cdf_ratio = cdf_normalized / np.max(cdf_normalized)
    # for i, val in enumerate(cdf_ratio):
    #     print (i, val)

    # Plot S channel histogram
    plt.subplot(133)
    plt.plot(cdf_normalized, color='orange')
    plt.hist(s_warped.flatten(), bins=256, color='b', range=[0, 256])
    plt.title('S Channel Histogram')

    plt.savefig(os.path.join(TMP_DIR, 'hist_hls.png'))
    plt.clf()


def color_thresh(img):
    """
    Convert RGB to HSL and threshold to binary image
    """
    
    # Step 1: Filter out pixels with strong reflection
    img = img.copy()
    blue_channel = img[:, :, 0].astype(np.float32)
    red_channel = img[:, :, 2].astype(np.float32)
    blud_red_diff = red_channel - blue_channel > 30
    

    # Step 2. Convert the image from RGB to HSL
    # For HSL
    # ref: https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html#color_convert_rgb_hls
    #   image format: (8-bit images) V ← 255⋅V, S ← 255⋅S, H ← H/2(to fit to 0 to 255)
    hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    h, l, s = cv2.split(hls_img)
    binary_output = np.zeros_like(l)

    # Step 3: Apply dynamic threshold on the S (Saturation) channel
    # Dynamically search saturation thres using saturation histogram of bird-eye image.
    s_warped, M, Minv = perspective_transform(s)
    sat_hist, bins = np.histogram(s_warped.flatten(), bins=256, range=[0, 256])

    # Calculate the cumulative distribution function (CDF) of the histogram
    cdf = sat_hist.cumsum()
    cdf_normalized = cdf / cdf.max()  # Normalize the CDF to the range [0, 1]
    bin_idxs = \
        np.where((cdf_normalized > args.sat_cdf_lower_thres)
                 & (cdf_normalized < 0.90))[0]
    sat_thres_min = np.argmin([sat_hist[idx]
                              for idx in bin_idxs]) + bin_idxs[0]
    sat_cond = ((sat_thres_min <= s) & (s <= 255))
    
    # for debug
    # print (bin_idxs)
    # for idx in bin_idxs:
    #     print ("{} => {}".format(idx, sat_hist[idx]))    
    # print ("sat_thres_min:", sat_thres_min)

    # Step 4: Apply dynamic threshold on the RGB "red" channel.
    # Reason to use red channel here because red channel values for yellow lane is significantl different from background's values
    red_channel = img[:, :, 2]  # red channel
    red_channel_warped, M, Minv = perspective_transform(red_channel)
    val_thres_min = np.percentile(
        red_channel_warped, args.val_thres_percentile)
    # val_mean = np.mean(red_channel_warped)
    val_cond = (val_thres_min <= red_channel) & (red_channel <= 255)
    
    bias = 15
    mean_val = np.mean(red_channel_warped)
    mean_val_cond = mean_val + bias <= red_channel
    
    # Step 5: Apply predefined hue threshold on image
    hue_cond = (hue_thres_min <= h) & (h <= hue_thres_max)

    # Step 6: Combine conditions to get final output
    binary_output[val_cond & sat_cond & hue_cond & blud_red_diff & mean_val_cond] = 1
    # binary_output[val_cond & hue_cond & blud_red_diff] = 1

    # Step 7: Closing small holes inside the yellow lane
    kernel = np.ones((5, 5), np.uint8)
    binary_output = cv2.morphologyEx(
        binary_output.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    ### visualization three channels result ###
    if args.vis_mode:
        vis_hls_hist(h, l, s)
        hist_hls = cv2.imread(os.path.join(TMP_DIR, 'hist_hls.png'))
        hist_hls = fixedAspectRatioResize(
            hist_hls, desired_width=hist_hls.shape[1]*3)
        imshow("hist_hls.png", hist_hls)

        vis_h = np.zeros_like(img)
        vis_s = np.zeros_like(img)
        vis_l = np.zeros_like(img)
        vis_val_mean = np.zeros_like(img)
        vis_h[hue_cond] = 255
        putText(vis_h, "hue result")
        vis_s[sat_cond] = 255
        putText(vis_s, "sat result")
        vis_l[val_cond] = 255
        putText(vis_l, "val result")
        vis_val_mean[mean_val_cond] = 255
        putText(vis_val_mean, "vis_val_mean")
        imshow("hls result", np.hstack([img, vis_h, vis_s, vis_l, vis_val_mean]))

    ### visualization for hue testing ###
    if args.vis_sat:
        OUTPUT_DIR = "./sat-test"
        pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        step = 5
        MAX_SAT = 255
        for i in range(0, MAX_SAT - step, step):
            mask = cv2.inRange(hls_img[:, :, 2], i, i + step)
            mask[mask > 0] = 255
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            vis = cv2.hconcat([img, mask])
            cv2.imwrite(os.path.join(OUTPUT_DIR, 'L{}.jpg'.format(i)), vis)

    if args.vis_hue:
        OUTPUT_DIR = "./hue-test"
        pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        step = 5
        MAX_HUE = 180
        for i in range(0, MAX_HUE - step, step):
            mask = cv2.inRange(hls_img[:, :, 0], i, i + step)
            mask[mask > 0] = 255
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            vis = cv2.hconcat([img, mask])
            cv2.imwrite(os.path.join(OUTPUT_DIR, 'L{}.jpg'.format(i)), vis)

    return binary_output


def get_matrix_calibration(img_shape,
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

def perspective_transform(img):
    """
    Get bird's eye view from input image
    """
    # 1. Visually determine 4 source points and 4 destination points
    # 2. Get M, the transform matrix, and Minv, the inverse using cv2.getPerspectiveTransform()
    # 3. Generate warped image in bird view using cv2.warpPerspective()

    # Define four points as (x, y) coordinates
    src_height, src_width = img.shape[:2]

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

    # keep same size as input image
    warped_img = cv2.warpPerspective(
        img, M, (dst_width, dst_height), flags=cv2.INTER_NEAREST)

    ### vis poly lines ###
    # concat = cv2.vconcat([raw_img, vis_dst])
    # if args.vis_mode:
    #     imshow("img", concat)

    return warped_img, M, Minv


def fixedAspectRatioResize(img, desired_height=None, desired_width=None):
    if desired_width:
        # Calculate the new height to maintain the aspect ratio
        height, width, _ = img.shape
        aspect_ratio = width / height
        desired_height = int(desired_width / aspect_ratio)

        # Resize the image
        return cv2.resize(img, (desired_width, desired_height))
    else:
        exit(1)
        return -1


def line_fit(binary_warped):
    """
    Find and fit lane lines
    """

    # Step 1. Use sliding window to find the base point
    height, width = binary_warped.shape
    nwindows = 15
    sliding_offset = 5
    margin = 70
    best_base_x = -1
    best_num_pixels = 0

    for basex in range(margin, width-margin, sliding_offset):
        left = basex - margin
        right = basex + margin
        total_num_pixels = cv2.countNonZero(
            binary_warped[-args.window_height:, left:right])

        if total_num_pixels > best_num_pixels:
            best_num_pixels = total_num_pixels
            best_base_x = basex

    if best_base_x == -1:
        return None

    # visualize
    # vis = cv2.cvtColor(binary_warped*255, cv2.COLOR_GRAY2BGR)
    # vis = cv2.rectangle(
    #     vis, (best_base_x - margin, height - args.window_height),
    #     (best_base_x + margin, height),
    #     (0, 0, 255))
    # imshow("vis", vis)

    # params setting
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    minpix = 200  # minimum number of pixels found by window
    color_warped = cv2.cvtColor(binary_warped*255, cv2.COLOR_GRAY2BGR)
    color_warped[color_warped > 0] = 255

    # Step 2. Use window to fit the lane
    basex = best_base_x
    lane_pts = []
    prev_basex_list = []
    i = 0
    while True:
        win_top = height - (i + 1) * args.window_height
        win_bottom = win_top + args.window_height

        # adjust base_x based on the slope of previous two basex
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
        if reach_boundary or len(window_nonzerox) < minpix:
            break

        # correct basex by average and use average (x, y) as way points
        basex = int(np.mean(window_nonzerox))
        basey = int(np.mean(window_nonzeroy))
        lane_pts.append([basex, basey])
        prev_basex_list.append(basex)

        i += 1
        
        # visualization
        color_warped = cv2.rectangle(
            color_warped, (basex - margin, win_top), (basex + margin, win_bottom), (0, 0, 255))
        # imshow("color_warped", color_warped)

    ### vis color_warped ###
    putText(color_warped, "warp & lanefit")

    # Step 3. Find the fitting polynomial
    lanex = [pt[0] for pt in lane_pts]
    laney = [pt[1] for pt in lane_pts]
    try:
        lane_fit = np.polyfit(laney, lanex, deg=2)

        ### vis lane points ###
        for x, y in zip(lanex, laney):
            color_warped = cv2.circle(color_warped, (x, y), 2, (0, 255, 0), -1)

        ### vis points nonzero ###
        # for x, y in zip(rightx, righty):
        #     color_warped = cv2.circle(color_warped, (x, y), 1, (0,255, 0), -1)
        # imshow("points", color_warped )

    except TypeError:
        print("Unable to detect lanes")
        return None

    ret = {}
    ret['vis_warped'] = color_warped
    ret['lane_fit'] = lane_fit
    ret['lanex'] = lanex
    ret['laney'] = laney
    return ret

def convert2CalibrationCoord(shape, lanex, laney, Minv):
    clb_M, clb_Minv = get_matrix_calibration(shape)
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
        
def run(img_path, fail_paths):
    global img_name
    img_name = img_path.strip('.png').split('/')[-1]
    print("img_path:", img_path)

    img = cv2.imread(img_path)

    # use HLS color space to filter
    ColorOutput = color_thresh(img)
    color_warped, M, Minv = perspective_transform(ColorOutput)
    ret = line_fit(color_warped)

    if ret is None:
        fail_paths.append(img_path)
        print("Fail to fit line")
        return
            
    convert2CalibrationCoord(img.shape[:2], ret['lanex'], ret['laney'], Minv)

    ### visulization all ###
    # SobelOutput = cv2.cvtColor(SobelOutput*255, cv2.COLOR_GRAY2BGR)
    ColorOutput = cv2.cvtColor(ColorOutput*255, cv2.COLOR_GRAY2BGR)
    concat = cv2.vconcat([img, ColorOutput, ret['vis_warped']])
    if args.vis_mode:
        imshow("warped", ret['vis_warped'])
        # imshow("concat", concat)
    if args.vis_output:
        cv2.imwrite(os.path.join(
            TMP_DIR, 'result_{}.png').format(img_name), concat)


if __name__ == '__main__':

    print('======= Initial parameters =======')
    params = []
    for key, val in vars(args).items():
        param = f"{key} => {val}"
        print(param)
        params.append(param)

    pathlib.Path(TMP_DIR).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(TMP_DIR, 'a_params.txt'), 'w') as f:
        f.write('\n'.join(params))

    fail_paths = []
    if args.specified_name:  # for single image file testing
        img_path = os.path.join(
            args.input_dir, '{}.png'.format(args.specified_name))
        run(img_path, fail_paths)
    else:  # loop through the image files under input_dir
        paths = sorted(os.listdir(args.input_dir))
        for i, path in enumerate(paths):
            if i == args.num_samples:
                break
            if not path.endswith('png'):
                continue

            img_path = os.path.join(args.input_dir, path)
            run(img_path, fail_paths)

    print("\n ----- {} failed images -----".format(len(fail_paths)))
    for i, path in enumerate(fail_paths):
        print('{}. {}'.format(i+1, path))
