### Modifying STLs for screen, battery, and Jetson housing

The default design consists of a microscope component and a separate deep learning component. The microscope STLs are fixed due to specific dimensions (such as focal lengths and tube lengths) to achieve the desired magnification. However, the deep learning component also houses a touch screen and battery packs, which are up to a high degree of customization specific to a build (such as using different sized battery packs and screens with different dimensions). 

To address this, we provide the openSCAD scripts that generate the STLs and made it modular and easy to customize so that the STLs can be adapted to these highly variable parts.

#### Using openSCAD

The scripts are located in `lcd_and_board.scad`. These scripts generate both the top and bottom half of the housing component. See the comments in the script for details.

#### Other options

Aside from directly modifying openSCAD, these parts were designed with relatively simple transforms in mind so that it is recognizable by CAD software, such as using the Recognize Features function of FeatureWorks as part of SolidWorks).
