![](https://img.shields.io/badge/SDK-v12.9.0-blue) <Please check version is the same as specified in requirements.txt>

# Actueel hoogtebestand Nederland (AHN) viewer
This sample app shows how to retrieve data from some GeoServer using the Web Map Service (WMS). 

When hosting this app, you can draw lines on a map, after which you can retrieve a graph of the AHN along this line.
An example interaction is shown below.

![
Inside the application you can draw a line on a map in the "Map" tab, 
and view the resulting AHN graph in the "AHN along line" tab
](source/images/sample-app-overview.gif "Sample app overview")

### Code usage
AHN specific code can be found in `app/functions/ahn.py`. The actual code that retrieves the data from the server is
VIKTOR independent, and hence can also be used without VIKTOR. There is no visualisation available in this file. 

Inside the application however, input and output are generated using VIKTOR.
