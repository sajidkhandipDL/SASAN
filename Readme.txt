Author: Sajid Khan
Annotation of datasets: Kainat Fareed
Version: 1.0
Last updated: 8th January 2023



1. 
Directory-->ASAN enhanced

ASAN dataset in un-collage and enhanced form are stored in the subdirectories created according to their class type. 

Note that the stored dataset is without ROI extraction.
2.  
Directory-->JSON samples
This directory contain some of the JSON samples obtained after annotation. These JSON files contain points that combinely
form the boundary of lesions. A single JSON file may contain points related to two or more than two lesions.

3. 
Python source code-->JSON2Binary.py
This is a python source code that loads all of the JSON files in "JSON samples" directory, convert them into binary images, and stores these binary 
images in the directory named "JSON 2 Masks"

4.
Directory-->MATLAB un-collage ASAN dataset
This directory contains three sample un-collage images along with a MATLAB script. User just need to provide the name of the un-collage image, it creates 
a subdirectory for that image, and will store the un-collage images in it.

5.
Directory-->Samples intensity images
This directory contains some sample images that are used by the python source file named "intensity2ROI.py"

6. 
Python source code-->intensity2ROI
In this file, you can load any of the top 5 trained semantic segmentation network shown in the Table of the manuscript. The loaded network will
then be used to extract ROI from the intensity images provided in the directory "Samples intensity images". Those top 5 trained models
are provided in directory "Saved Seg Models"

7.  
Directory-->Saved Seg Models
This directory must contains h5 files for the top five trained semantic segmentation models. Because of GitHub storage limitations, we weren't 
able to push them to the repository. You are requested to download the h5 files from the following url
https://drive.google.com/drive/folders/1cxXE3cqNfZr5_kfv2vKQgXDJSq2kNxRc?usp=share_link

8. 
Directory-->Train Semantic Segmentation
This directory contains four python files that are as below
i.  "augment.py"  used to augment the ground truths
ii. "Call2SegmentationModule.py" it has six nested loops that perform hundreds of experiments. It ensures that if an experiment
is already performed and the h5 file related to it is already available, it will simply skip that experiment.
iii.  "SegmentationModule.py" It has implementations of semantic segmentation. This file is called "Call2SegmentationModule.py" at
 each loop iteration.
iv.  "UScratchASAN.py" This file has the implementation of UNet without using the framework segmentation_models. 


For any query, please contact sajidkhandip2022@gmail.com