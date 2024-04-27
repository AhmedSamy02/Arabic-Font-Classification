import cv2
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

 
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'  # your path may be different

imgPath = '8.jpeg'

def binarizeImage(RGB_image):

  image = rgb2gray(RGB_image)
  threshold = threshold_otsu(image)
  print('threshold:', threshold)
  bina_image = image < threshold
  return bina_image

def rotate_image(image, angle):
    # Get image dimensions
    h, w = image.shape[:2]
    
    # Calculate rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    
    # Perform rotation
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h),flags=cv2.INTER_CUBIC)
    
    return rotated_image

def hough_transforms(image):
    # Read the image
    # image = cv2.imread(image_path)
    
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
    
    for angle, dist in zip(angles, dists):
    # Check if sin(angle) is close to zero (vertical line)
        if np.isclose(np.sin(angle), 0):
            # Handle vertical lines separately
            x = int(dist)
            cv2.line(lined_image, (x, 0), (x, lined_image.shape[0]), (0, 255, 0), 2)
        else:
            # Calculate y0 and y1 for non-vertical lines
            y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
            y1 = (dist - lined_image.shape[1] * np.cos(angle)) / np.sin(angle)
            # Ensure y0 and y1 are finite
            if np.isfinite(y0) and np.isfinite(y1):
                cv2.line(lined_image, (0, int(y0)), (lined_image.shape[1], int(y1)), (0, 255, 0), 2)

    cv2.imwrite('outlined.jpg', lined_image)

    
    # Print the detected angles
    print('Number of lines detected:', len(angles))
    print('Detected angles(rad):', angles)

    angles = angles * (180 / np.pi)
    # angles = angles[np.round(angles) != 90]
    rotation_angle = mode(angles).mode if len(angles)>0  else 0

    # angles = angles[(np.round(angles) < 89) | (np.round(angles) > 91)]

    print('Detected angles(degree):', angles)

    rotation_angle = mode(angles).mode if len(angles)>0  else 0

    if (rotation_angle < 15 ):
        rotation_angle = 0    
    else:
        rotation_angle = rotation_angle -90
    # Rotate the image using the first detected angle
    print('rotation angle:', rotation_angle)
    rotated_image = rotate_image(image,  rotation_angle)
    
    # Save the rotated image
    cv2.imwrite('hough_out.jpeg', rotated_image)


def pytesseract_orientation(image_path):
    osd = None
    image = cv2.imread(image_path)

    try:
        osd = pytesseract.image_to_osd(image_path, config=' --psm 0 -c min_characters_to_try=5', output_type=Output.DICT)
    except Exception as e:
        print("Error: " + str(e))
        cv2.imwrite('correct_orient.jpeg', image)
        return None
        # osd = pytesseract.image_to_osd(image_path, config=' --psm 0 -c min_characters_to_try=1', output_type=Output.DICT)
    # osd = pytesseract.image_to_osd(image_path, config=' --psm 0 -c min_characters_to_try=5', output_type=Output.DICT)
    print("[OSD] " + str(osd))
    rotated = rotate_image(image, angle=osd["orientation"])
    # return rotated
    cv2.imwrite('correct_orient.jpeg', rotated)


if __name__ == '__main__':
     
#   pytesseract_orientation(imgPath)
#   hough_transforms(pytesseract_orientation(imgPath))
  outout_image=filters.median(cv2.imread(imgPath))
  cv2.imwrite('filtered.jpeg', outout_image)

#   hough_transforms(image=cv2.imread(imgPath))
  hough_transforms(image=cv2.imread('filtered.jpeg'))
  pytesseract_orientation('hough_out.jpeg')

  temp = io.imread('correct_orient.jpeg')

  bina_img = binarizeImage(temp)
  plt.savefig('binarized.jpeg')


