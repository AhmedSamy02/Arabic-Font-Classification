import cv2
import os
kernel = cv2.getStructuringElement(cv2.MORPH_ERODE, (10,4))

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
def threshold_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray_image = cv2.GaussianBlur(gray_image, (3, 3), 1)
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
def find_contours(thresholded_image,dilated_image,path,count):
    cnts = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    length,width= dilated_image.shape
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    # temp_image = image.copy()
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)

        ## Remove bounding
        if x==0 and y==0:
            continue
        elif x+w==width and y+h==length:
            continue
        elif x+w ==width and y==0:
            continue
        elif x==0 and y+h==length:
            continue
        elif w<120 or h<25:
            continue
        # cv2.rectangle(image, (x, y), (x + w, y + h), (255,255,0), 2)
        cv2.imwrite(path+str(count)+'.jpeg', thresholded_image[y:y+h,x:x+w])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        count+=1
    return count
def crop_image(input_directory,output_directory):
    images = load_images_from_directory(input_directory)
    count=0
    imageCount = 1
    for image in images:
        print("Processing image "+str(imageCount))
        imageCount+=1
        thresholded_image = threshold_image(image)
        dilated_image = cv2.dilate(thresholded_image, kernel, iterations=4)
        count= find_contours(thresholded_image,dilated_image,output_directory,count)
