# Arabic-Font-Recognition <img align= center width=70px height=70px src="https://github.com/YaraHisham61/Arabic-Font-Recognition/assets/88517271/5647920b-c205-4aac-9d75-a6d0f61a1a62">

## <img align= center width=50px height=50px src="https://github.com/AhmedSamy02/Adders-Mania/assets/88517271/dba75e61-02dd-465b-bc31-90907f36c93a"> Table of Contents

- [Overview](#overview)
- [Project Pipeline](#pipe)
- [Final Results](#final)
- [Contributors](#contributors)
- [License](#license)


## <img src="https://github.com/AhmedSamy02/Adders-Mania/assets/88517271/9ed3ee67-0407-4c82-9e29-4faa76d1ac44" width="50" height="50" /> Overview <a name = "overview"></a>
Given an image containing a paragraph written in Arabic, the system classifies the paragraph into one of four fonts (from 0 to 3) with classical machine learning techniques.
| Font Code 	| Font Name |
|:-------|:-------------------|
| 0	|  Scheherazade New |
| 1 | Times New Roman|
| 2 |	Lemonada |
| 3 |	 IBM Plex Sans Arabic |


## <img src="https://github.com/YaraHisham61/Arabic-Font-Recognition/assets/88517271/a1f4f29e-84dd-4a24-871c-8a9118265430" width="30" height="30" /> Project Pipeline <a name = "pipe"></a>

![image](https://github.com/YaraHisham61/Arabic-Font-Recognition/assets/88517271/d974a850-a2a3-4cc1-89ae-cc498c38255f)
### Preprocessing Module
This preprocessing module is designed to enhance the quality of images containing text before further analysis or processing. Here's a breakdown of the module:
1.	Salt and Pepper Noise Detection and Removal:
-	The detect_salt_and_pepper function assesses if the image contains salt and pepper noise.
-	If detected, it applies a median filter (median_filter) to reduce the noise.
2.	Binarization:
-	The binarizeImage function converts the grayscale image to a binary image, making text clearer for extraction.
-	It ensures the image is in black text on a white background.
3.	Hough Line Transform:
-	The hough_transforms function detects lines in the binary image using the Hough line transform.
-	It rotates the image to align detected lines horizontally.
4.	Text Orientation Correction:
-	The pytesseract_orientation function uses Tesseract OCR to detect the text orientation.
-	It rotates the image based on the detected orientation for proper alignment.
5.	Image Preprocessing Pipeline:
-	The preprocess function orchestrates the entire preprocessing pipeline:
-	Loads the image.
-	Detects salt and pepper noise and removes it if present.
-	Binarizes the image.
-	Applies the Hough transform to align text horizontally.
-	Corrects the text orientation.
-	Saves the preprocessed image.
6.	Segmentation of Text
-	The find_contours function identifies contours within the thresholded and dilated image.
-	Contours represent the boundaries of distinct objects or regions within the image.
-	It filters out small or insignificant contours based on width and height thresholds.
-	For each valid contour, it creates a bounding box around the region of interest and saves it as a separate image file in a specified output directory 
-	The function returns the count of extracted regions.
### Feature Extraction/Selection Module
We tried the following approaches: 
-	Horizontal and vertical Histogram
-	Entropy
-	HoG
-	SIFT
-	Local Phase Quantization (LPQ)
After comparing results of the listed above approaches we decided to use LPQ as it yields the best results 

### Model Selection/Training Module
We tried the following approaches:
-	KNN
-	SVM
-	Decision Tree
-	Random Forest 
After comparing results of the listed above approaches we decided to use Random Forest as it yields the best results with LPQ
### Performance Analysis Module
Testing the above classifiers we got the following results:
-	Total Accuracy of KNN = 91.54481842707052 %
-	Total Accuracy of Decision Tree = 93.34605637273302 %
-	Total Accuracy of SVM = 71.19171382008213 %
-	Total Accuracy of Random Forest = 98.80024797790851 %

## <img src="https://github.com/YaraHisham61/Arabic-Font-Recognition/assets/88517271/f42b863b-c284-4db9-bb59-00b9062f0f3d" width="50" height="50" /> Final Results <a name = "final"></a>
Our System scored an accuracy of 98% on test data.


## <img src="https://github.com/YaraHisham61/OS_Scheduler/assets/88517271/859c6d0a-d951-4135-b420-6ca35c403803" width="50" height="50" /> Contributors <a name = "contributors"></a>
<table>
  <tr>
   <td align="center">
    <a href="https://github.com/AhmedSamy02" target="_black">
    <img src="https://avatars.githubusercontent.com/u/96637750?v=4" width="150px;" alt="Ahmed Samy"/>
    <br />
    <sub><b>Ahmed Samy</b></sub></a>
    </td>
   <td align="center">
    <a href="https://github.com/kaokab33" target="_black">
    <img src="https://avatars.githubusercontent.com/u/93781327?v=4" width="150px;" alt="Kareem Samy"/>
    <br />
    <sub><b>Kareem Samy</b></sub></a>
    </td>
   <td align="center">
    <a href="https://github.com/nancyalgazzar" target="_black">
    <img src="https://avatars.githubusercontent.com/u/94644017?v=4" width="150px;" alt="Nancy Ayman"/>
    <br />
    <sub><b>Nancy Ayman</b></sub></a>
    </td>
   <td align="center">
    <a href="https://github.com/YaraHisham61" target="_black">
    <img src="https://avatars.githubusercontent.com/u/88517271?v=4" width="150px;" alt="Yara Hisham"/>
    <br />
    <sub><b>Yara Hisham</b></sub></a>
    </td>
  </tr>
 </table>

  ## <img src="https://github.com/YaraHisham61/Architecture_Project/assets/88517271/c4a8b264-bf74-4f14-ba2a-b017ef999151" width="50" height="50" /> License <a name = "license"></a>
> This software is licensed under MIT License, See [License](https://github.com/YaraHisham61/Arabic-Font-Recognition/blob/master/LICENSE)
