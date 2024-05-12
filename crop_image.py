import cv2
import os
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE , (10,4))

def load_images_from_directory(directory):
    images = []
    count=0
    for filename in os.listdir(directory):
        if filename.endswith(".jpeg") or filename.endswith(".png"):
            count+=1
            img = cv2.imread(os.path.join(directory, filename))
            if img is not None:
                images.append(img)
    print("Got "+str(count)+" images")
    return images
def threshold_image(image,median_flag):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray_image = cv2.GaussianBlur(gray_image, (3, 3), 1)
    if median_flag == True:
        gray_image = cv2.medianBlur(gray_image,3)
    _, thresholded_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    l,w = thresholded_image.shape
    l=l//2
    w=w//2
    count=0
    if thresholded_image[l-1,w-1]!=0:
        count+=1
    if thresholded_image[10,10]!=0:
        count+=1
    if thresholded_image[10,w-1]!=0:
        count+=1
    if thresholded_image[l-1,10]!=0:
        count+=1
    if count>=2:
        thresholded_image = cv2.bitwise_not(thresholded_image)
    return thresholded_image
def delete_files_in_directory(directory_path):
    try:
        with os.scandir(directory_path) as entries:
            for entry in entries:
                if entry.is_file():
                    os.unlink(entry.path)
        print("All files deleted successfully.")
    except OSError:
        print("Error occurred while deleting files.")
def find_contours(thresholded_image,dilated_image):
    cnts = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    length,width= dilated_image.shape
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cropped_images = []
    # temp_image = image.copy()
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)

        ## Remove bounding
        if len(cnts)!=1:
            if x==0 and y==0:
                continue
            elif x+w==width and y+h==length:
                continue
            elif x+w ==width and y==0:
                continue
            elif x==0 and y+h==length:
                continue
        if w<120 or h<25:
            continue
        # cv2.rectangle(image, (x, y), (x + w, y + h), (255,255,0), 2)
        # cv2.imwrite(path+str(count)+'.jpeg', thresholded_image[y:y+h,x:x+w])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cropped_images.append(thresholded_image[y:y+h,x:x+w])
    return cropped_images
def crop_image(image,dilation_iterations,median_flag):
    # images = load_images_from_directory(input_directory)
    thresholded_image = threshold_image(image,median_flag)
    dilated_image = cv2.dilate(thresholded_image, kernel, iterations=dilation_iterations)
    # cv2.imshow('Dilated Image',dilated_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cropped_images = find_contours(thresholded_image,dilated_image)
    return cropped_images
