# Track Helper: Detection and Navigation of Route Panorama System based on YOLOV4
迷航小幫手—YOLOv4物件偵測與導航系統

# Abstract  
Most people go out instead of using traditional maps to view paths, but instead use satellite navigation for positioning and destination search. However, if the location range is too small (e.g. a school), the navigation system can only show which school the user is currently in, while they cannot determine which floor or building of the school.  

Therefore, we developed the "Trek Helper" system, which uses the self-photographed building images of the National University of Education as the main training dataset to solve the detailed position recognition that cannot be achieved by general satellite navigation. In this system. The user can open the lens to take pictures, achieve the purpose of determining the exact location, detect the user's current location. And after the user enters the destination requirements, the system will analyze and plan the route, give a clear travel route. Different from the traditional campus map or playground sightseeing guide map, this system is an extension of satellite navigation and the basis for future AI intelligent tour guides.  

# System Introduction

1. **Background**  
    In recent years, technology has become more and more developed, most people no longer use traditional maps to view the path when traveling, but instead use satellite navigation systems for positioning and purpose navigation. But for satellite navigation systems, if people's location range is too small (such as a school, an amusement park area), the navigation system would show more positioning results that can not clearly identify the user's exact location, and can only locate which school or which amusement park the user is currently in. At this time, it is not possible to tell which floor or building the user is on the school or amusement park.    
    As shown in the figure below, people who want to go to **Sports Center** may accidentally be led to an adjacent but incorrect position. Though the location indicated below is close to the place where we want to go, it is practically impossible to get to Sports Center from there.  
    <img src="./images/00.jpg" width="50%">  
    As a result of these reasons, most people will not be able to determine their exact location, and after arriving at the school or amusement park, they can only use a traditional campus map or sightseeing guide map to see their current location and determine how to get to their destination. When you are in a hurry or have a hurry, it is easy to be late because you don't know how to walk around the campus or amusement park.    
    For the reasons above, we have developed a "Trek Helper System" based on the theme of environmental images of the campus of National Taipei University of Education, showing a system that can directly take pictures of the surrounding environment and solve the detailed position recognition that cannot be achieved by ordinary satellite navigation. In this system, the user can open the lens to take pictures to achieve the requirements of determining the exact position of the use. After detecting the user's current location, the system will also analyze the current location situation, route planning, and give a clear travel route after the user enters the needs of the purpose. Different from the traditional campus map or playground sightseeing guide map, this system is an extension of satellite navigation and will be the the basis for the AI intelligent tour guides in the future.  


2. **Introduction to System Functionality**  
    In this system (as shown in Figure 1), the users can **determine their exact position** by opening the lens to take pictures. After detecting the user's current location, the system will also analyze the current location situation, carry out **route planning**, and give a clear travel route after the user enters the demand of the purpose.
    <img src="./images/01.jpg" width="80%">

    **Illustration**:  
    (A) Take a random photo at their location and upload an image.  
    (B) Press the button, and the system performs position recognition.  
    (C) The results are displayed on a campus map and the user's current location is indicated.  
    (D) Select the destination location in the drop-down menu.  
    (E) The system plans the route and displays the best route to travel.  
    (F) If you want to plan your route directly, change your departure location directly.  

3. **Features**  
    The main purpose of this system is to **identify the street view of areas that cannot be clearly identified by satellite navigation, taking pictures of mobile devices to determine the user's current location.** In addition, the system also provides a **navigation** function, which allows users to automatically navigate through the selection of destination location through route planning technology, so as to help visitors in need to visit the campus more smoothly.

4. **Technical Skills**
    - Python
    - YOLOv4 Object Detection
    - OS: Windows, Linux/Ubuntu
    - Image Labeling
    - HTML / CSS / Javascript
    - Web frame: Flask

5. **Users**  
    This system is mainly designed for visitors who are **not familiar with the campus of National Taipei University of Education**, including: new students, transfer students, on-campus visitors, visitors with official needs, etc.  

6. **Development Method**  
    Step.01  LabelImage  
        <img src="./images/02.png" width="80%">

    Step.02  Model Training: the process is shown in the following figure
        <img src="./images/03.png" width="80%">

    Step.03  Model Training Results and Tests
        <img src="./images/04.png" width="80%">

    Step.04  Rrocess of Image Recognition
        <img src="./images/05.png" width="80%">
  
    Step.05  Flask Runner (Example)
        <img src="./images/06.png" width="80%">  
  
    Step.06  Flask: Web Image Upload
        <img src="./images/07.png" width="80%">  
  
    Step.07  Landmark and Route Navigation
        <img src="./images/08.jpg" width="80%">  


7. **Conclusion**  
    In this project, we used the images of the National Taipei University of Education as the main training data. However, in addition to be used as a campus identification and navigation system, this work can also use the same technology and different image data to **be applied in most amusement park areas in Taiwan**.  
    Taking the research of this project team as the basis for subsequent application, this technical architecture can be applied to major domestic attractions such as Luofu Village, Jiu Ethnic Culture Village, Xitou Forest Recreation Area, etc. As the basis for AI guidance, when satellite navigation is "not so reliable", this technical architecture can be used for implementation: using the images of the amusement park/sightseeing spot as new training data, building models and implementing systems, so **it can solve most of the problems caused by poor satellite signals and positioning errors**.  
    In the future, we also hope to continue in-depth research on the computer vision, deep learning and image processing technologies used in this system. Combined with the IoT, cloud computing, blockchain and other technologies, we can analyze object detection and image processing problems from a more professional perspective, conduct intelligent control and satellite detection research, which extends the system to more related applications.

# How to train a custom yolov4 model  
## 0. Initial Download  
- official darknet(https://github.com/AlexeyAB/darknet)
- official pre-trained weights(https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137)
## 1. Getting Start - GPU Settings
1. Download official files
2. Set up GPU Configuration

    You can set up the GPU configuraiton by following instructions.
    ```
    !sed -i 's/OPENCV=0/OPENCV=1/' Makefile
    !sed -i 's/GPU=0/GPU=1/' Makefile
    !sed -i 's/CUDNN=0/CUDNN=1/' Makefile
    !sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile
    !sed -i 's/LIBSO=0/LIBSO=1/' Makefile
    ```
3. build up darknet environment
    ```
    !make
    ```
  
## 2. Data Preperation
1. Clean the data and cfg folders first except the labels folder in data which is required
2. Unzip the obj.zip dataset and its contents so that they are now in /darknet/data/ folder
3. Copy the yolov4-custom.cfg file so that it is now in /darknet/cfg/ folder 

    For the configuration file, we need to change some parameters inside.  
  
    ```
    # yolov4-custom.cfg
    [net]
    batch = 64
    subdivisions= 64
    max_batches = (category)*2000, less than 4000
    steps = max_batches * 0.8, max_batches * 0.9
    width=416
    height=416
    ...
    ...
    [yolo]
    classes = n (number of categories)
    ...
    ...
    [yolo]
    classes = n (number of categories)
    ...
    ...
    [yolo]
    classes = n (number of categories)
    ...
    ...
    ```
  
4. Copy the obj.names and obj.data files from your drive so that they are now in /darknet/data/ folder 

    ```
    # obj.names
    administ
    library
    audio
    science
    tower
    ...
    ...
    ```
  
    ```
    # obj.data
    classes= n (number of categories)
    train  = data/train.txt
    valid  = data/test.txt
    names = data/obj.names
    backup = (the path for you to save the trained weights)
    ```
  
5. Transform data
  
    For obj dataset, you can run process.py.
    ```
    # train.txt
    image_path1 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
    image_path2 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
    ...
    ...
    ```
6. After all, you can check the file dictory like this:
    ```
    project
    │   README.md
    │
    └───darknet
    │   │   process.py
    │   │   yolov4.conv.137
    │   │   ...
    │   │
    |   └───cfg
    |   |   |   yolov4-custom.cfg
    |   |   |
    |   |
    │   └───data
    |       |   obj.data
    |       |   obj.names
    |       |   train.txt
    |       |   test.txt
    |       |
    │       └───labels
    |       |   |   ...
    |       |   |
    |       |
    |       └───obj
    |           |   0001.jpg
    |           |   0002.jpg
    |           |   ...
    |           |
    |
    └───flask
    |   └───src
    |       │   app.py
    |       │   
    |       └───scripts
    |       |   |   javascript.html
    |       |   |
    |       |
    |       └───static
    |       |   |   background1.png
    |       |   |   background.png
    |       |   |   favicon.ico
    |       |   |   map.jpg
    |       |   |
    |       |   └───css
    |       |   |   |   bootstrap.css
    |       |   |   |   main.css
    |       |   |   |
    |       |   |
    |       |   └───output
    |       |   |   |   administ.jpg
    |       |   |   |   audio.jpg
    |       |   |   |   ...
    |       |   |   └───administ
    |       |   |   |   |   ...
    |       |   |   |   |
    |       |   |   |
    |       |   |   ...
    |       |   |
    |       |   └───uploads
    |       |       |   scan.jpg
    |       |       |
    |       |
    |       └───templates
    |           |   index.html
    |           
    └───mask_test_images
    |   |   scan.jpg
    |   |
    |
    └───training
        |   yolov4-custom_last.weights
        |   yolov4-custom_best.weights
        |
    ```

## 3. Train Custom-Detector
1. Train your custom detector!

    You can uncomment %%capture below if you run into memory issues or your Colab is crashing.  
  
    ```
    # %%capture
    !./darknet detector train data/obj.data cfg/yolov4-custom.cfg yolov4.conv.137 -dont_show -map
    ```
    
2. Check Performance

    You can check the mAP for all the saved weights to see which gives the best results ( xxxx here is the saved weight number like 4000, 5000 or 6000 snd so on )
  
    ```
    !./darknet detector map data/obj.data cfg/yolov4-custom.cfg /home/jcwang/YOLOV4_TRAINING/yolov4_elevenclasses/training/yolov4-custom_best.weights -points 0
    ```
  
## 4. Test with Images
1. Set your custom cfg to test mode
  
    ```
    !sed -i 's/batch=64/batch=1/' yolov4-custom.cfg
    !sed -i 's/subdivisions=16/subdivisions=1/' yolov4-custom.cfg
    ```
2. Test for only one image
  
    ```
    !./darknet detector test data/obj.data cfg/yolov4-custom.cfg ../training/yolov4-custom_best.weights ../mask_test_images/00073.jpg -thresh 0.3 
    ```
    
3. Test for test.txt images
  
    ```
    !./darknet detector test data/obj.data cfg/yolov4-custom.cfg ../training/yolov4-custom_best.weights -ext_output -dont_show < ../darknet/data/test.txt> result.txt -thresh 0.3
    ```

# How to build up the Flask System
## 0. Installing Flask
To install Flask, run the following command:  
    
```
pip install flask
```
## 1. Design the Flask API
```
# app.py
@app.route('/')
def index():
    return render_template('index.html')
...
...
@app.route("/forward/", methods=['POST'])
def move_forward():
    # Scanning picture which user uploads

    os.chdir('../../darknet')

    os.system('./darknet detector test data/obj.data cfg/yolov4-custom.cfg ../training/yolov4-custom_best.weights ../flask/src/static/uploads/scan.jpg -thresh 0.3')
    os.system('./darknet detector test data/obj.data cfg/yolov4-custom.cfg ../training/yolov4-custom_best.weights -ext_output -dont_show < ../darknet/data/scan.txt> result.txt -thresh 0.3')
    ...
    ...
if __name__ == "__main__":
app.run(debug=True)
```
    
    
```
# index.html
<html>

<head>
    <meta charset="utf-8">
    <link rel="shortcut icon" href="../static/favicon.ico">
    <title>迷航小幫手</title>
    <style>
        .upload {
    ...
    ...
```
  
  
## 2. Run the System

1. Run the application using the **python app.py** command.  
  
    ```
    python app.py
    ```
2. Once the application is running the output will be something like this:  
  
    ```
     * Serving Flask app 'app'
     * Debug mode: on
     * Running on http://127.0.0.1:5000
     * Restarting with stat
     * Debugger is active!
     * Debugger PIN: 134-693-687
     ```
   
3. Open a browser and type in the URL http://127.0.0.1:5000/, then you will receive the system.

# Reference
1. https://github.com/kiyoshiiriemon/yolov4_darknet
2. https://github.com/Tianxiaomo/pytorch-YOLOv4
3. https://github.com/andy6804tw/yolov4-darknet
