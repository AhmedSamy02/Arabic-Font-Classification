import re
import cv2
import os
import numpy as np
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
import pytesseract
from pytesseract import Output
from scipy.stats import mode
from scipy.ndimage import median_filter
from skimage import  filters , io
import matplotlib.pyplot as plt
from skimage.filters import gaussian

 
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'  # your path may be different

# imgPath = '899.jpeg'
import cv2
import numpy as np

def detect_salt_and_pepper(image_path, threshold=0.01):
    # Load the image
    # image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image_path, cv2.COLOR_BGR2GRAY)

    # Apply median blur to reduce noise
    blurred_image = median_filter(gray_image, 3)
    # cv2.imwrite('blurred_imagesssssssssssss.jpeg', blurred_image)
    # Compute the absolute difference between the original and blurred image
    diff_image = cv2.absdiff(gray_image, blurred_image)

    # Threshold the difference image to find the salt and pepper noise
    _, thresholded_image = cv2.threshold(diff_image, 30, 255, cv2.THRESH_BINARY)

    # Count non-zero pixels in the thresholded image
    non_zero_pixels = np.count_nonzero(thresholded_image)

    # Calculate the ratio of noisy pixels to total pixels
    height, width = thresholded_image.shape
    total_pixels = height * width
    noise_ratio = non_zero_pixels / total_pixels

    # Check if the noise ratio exceeds the threshold
    if noise_ratio > threshold:
        return True  # Salt and pepper noise detected
    else:
        return False  # No salt and pepper noise detected


def binarizeImage(imgPath):

  # Load the image
    # image = cv2.imread(imgPath)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(imgPath, cv2.COLOR_BGR2GRAY)

    # Threshold the image to get a binary image
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    count=0
    if binary_image[-1,-1]!=0:
        count+=1
    if binary_image[0,0]!=0:
        count+=1
    if binary_image[0,-1]!=0:
        count+=1
    if binary_image[-1,0]!=0:
        count+=1
        
    # print(count)
    if count < 2:
        binary_image = cv2.bitwise_not(binary_image)
    # Convert the image to black text on white background
    black_text_on_white = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    # Save or display the resulting image
    # cv2.imwrite('binary_img.jpeg', ~black_text_on_white)
    return black_text_on_white

def rotate_image(image, angle):
    print('angle:', angle)
    # Get image dimensions
    # image = cv2.imread(image_path)
    # cv2.imwrite('befor_hough_out.jpeg', image)
    h, w = image.shape[:2]
    # Calculate rotation matrix
    background_color =tuple( image[0, 0])
    rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    # Perform rotation
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    # Create a mask for the rotated image
    mask = np.ones_like(rotated_image) * background_color
    # Overlay the rotated image on the mask
    result = np.where(rotated_image == background_color, mask, rotated_image)
    return result

def hough_transforms(image) :
    # Read the image
    # image = cv2.imread(image)
    
    # Convert image to grayscale

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    thresh = cv2.GaussianBlur(gray, (11, 11), 0)
    
    # Detect edges using Canny
    edges = canny(thresh)

    # Define tested angles for Hough transform
    tested_angles = np.deg2rad(np.arange(0, 180.0))
    
    # Perform Hough transform
    h, theta, d = hough_line(edges, theta=tested_angles)
    
    lined_image = np.copy(image)
    # Find peaks in Hough space
    accum, angles, dists = hough_line_peaks(h, theta, d, num_peaks=10)
    # for angle, dist in zip(angles, dists):
    # # Check if sin(angle) is close to zero (vertical line)
    #     if np.isclose(np.sin(angle), 0):
    #         # Handle vertical lines separately
    #         x = int(dist)
    #         cv2.line(lined_image, (x, 0), (x, lined_image.shape[0]), (0, 255, 0), 2)
    #     else:
    #         # Calculate y0 and y1 for non-vertical lines
    #         y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
    #         y1 = (dist - lined_image.shape[1] * np.cos(angle)) / np.sin(angle)
    #         # Ensure y0 and y1 are finite
    #         if np.isfinite(y0) and np.isfinite(y1):
    #             cv2.line(lined_image, (0, int(y0)), (lined_image.shape[1], int(y1)), (0, 255, 0), 2)
    # cv2.imwrite('outlined.jpg', lined_image)
    # Print the detected angles
    # print('Number of lines detected:', len(angles))
    # print('Detected angles(rad):', angles)
    angles = angles * (180 / np.pi)
    # angles = angles[np.round(angles) != 90]
    rotation_angle = mode(angles).mode if len(angles)>0  else 0
    # print('Detected angles(degree):', angles)
    if len(set(angles)) == len(angles) and len(angles) > 1:
        rotation_angle = 0
    if (rotation_angle < 15 ):
        rotation_angle = 0    
    else:
        rotation_angle = rotation_angle - 90
    # Rotate the image using the first detected angle
    # print('rotation angle:', rotation_angle)
    # cv2.imwrite('befor_befor_hough_out.jpeg', image)
    rotated_image = rotate_image(image,  rotation_angle)
    # Save the rotated image
    # cv2.imwrite('hough_out.jpeg', rotated_image)

    return rotated_image


def pytesseract_orientation(image_path):
    osd = None
    try:
        osd = pytesseract.image_to_osd(image_path, config=' --psm 0 -c min_characters_to_try=10', output_type=Output.DICT)
    except Exception as e:
        print("Error: " + str(e))
        return image_path

    print("[OSD] " + str(osd))

    if osd["script"]  != "Arabic" and osd["orientation_conf"] <=3:
        # cv2.imwrite('correct_orient.jpeg', image_path)
        return image_path
    # For other orientations, rotate by the detected angle
    else:
        rotated = rotate_image(image_path, angle=osd["orientation"])
        # cv2.imwrite('correct_orient.jpeg', rotated)
        return rotated

def preprocess(input_path, output_path):
    image = cv2.imread(input_path)
    salt_and_pepper_present = detect_salt_and_pepper(image)
    if salt_and_pepper_present == True :
        output_image = filters.median(image)
    else : 
        output_image = image
    # cv2.imwrite('after_median.jpeg', output_image)
    binary_image = binarizeImage(output_image)
    # cv2.imwrite('after_color.jpeg', binary_image)
    hough_out = hough_transforms(image=binary_image)
    # cv2.imwrite('after_hough.jpeg', hough_out)
    correct_orient = pytesseract_orientation(hough_out)
    cv2.imwrite(output_path, correct_orient)

def process_images_in_folder(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through files and subfolders in the input folder
    for root, _, files in os.walk(input_folder):
        # Create corresponding subfolder in the output folder
        relative_path = os.path.relpath(root, input_folder)
        output_subfolder = os.path.join(output_folder, relative_path)
        os.makedirs(output_subfolder, exist_ok=True)

        # Process files in the current folder
        for filename in files:
            # Check if the file is an image
            if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
                # Get the input and output paths
                input_path = os.path.join(root, filename)
                output_path = os.path.join(output_subfolder, filename)
                print(f"Processing {input_path}...")
                # Process the image
                preprocess(input_path, output_path)


    # preprocess('352.jpeg', 'output.jpg')import re
