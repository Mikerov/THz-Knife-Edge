# THz-Knife-Edge

This is a Python GUI App for data analysis of THz-TDS knife-edge experiments. The app provides functionality to select a directory with
THz-TDS measurements at different positions of the knife. The app will extract positions of the knife from the file names. The user has four options of which parameter will be used for beam radius calculation: maximum value, peak-to-peak, squared peak-to-peak or area under the curve (intensity). The radio button "in mm?" should be ckecked if the x-positions are in mm. If some arbitrary scale is used, one can select the factor, e.g., factor 2 means that each x-position in the file name will be multiplied by 2 mm. The app also provides the option to select the range of THz signal in the time domain that will be used to extract the information.

![GUI Screen](Capture.PNG?raw=true "Title")


This code was written as part of the work as research associate in AG Koch.