<h1>Real-Time Unauthorized Construction Detection and Area Assessment Using Drones</h1>
Overview
This project utilizes Mask R-CNN for detecting and segmenting unauthorized construction areas from drone images and calculates their areas using Geospatial Data Integration (GDI). Additionally, Time-of-Flight (ToF) sensors are used to validate the depth and volume of the detected structures.

 " <a href="https://youtu.be/7WgU7KTvziM?si=r9oevuY64mlqoJZm" target="_blank">Watch the Project Demo on YouTube</a>

How It Works
1. Detection and Segmentation Using Mask R-CNN
The project employs the Mask R-CNN model trained on a custom dataset of drone images to segment areas of unauthorized construction.

Steps for Area Detection:
The segmented regions are extracted as binary masks.
These masks are georeferenced to associate them with real-world spatial coordinates.
Using Geospatial Data Integration (GDI) tools, the pixel area of the mask is converted into real-world units, such as square meters.

2. Validation with Time-of-Flight (ToF) Sensors
The ToF sensors are employed to measure the depth and assess the volumetric details of unauthorized constructions.

Steps for Validation:
Depth Analysis: Accurately measures the height of structures.
This additional step ensures reliable volumetric calculations and compliance validation.
Features
Real-Time Detection: Processes drone images instantly for unauthorized construction segmentation.
High Accuracy: Leverages Mask R-CNN for precise segmentation.
Area Calculation: Converts segmented regions into real-world measurements using geospatial tools.
Volumetric Validation: Uses ToF sensors for depth analysis, enhancing detection accuracy.
Requirements
Hardware
A drone equipped with a high-resolution camera.
Time-of-Flight (ToF) sensors for depth measurement.
Software
Python
TensorFlow
OpenCV
Geospatial tools such as QGIS or GDAL
Installation
bash
Copy code

# Navigate to the directory  
cd Real-Time-Unauthorized-Construction-Detection-and-Area-Assessment-Using-Drones  

Applications
Real-time monitoring of unauthorized construction in urban and rural areas.
Compliance checks for building regulations.
Land management and resource planning.



