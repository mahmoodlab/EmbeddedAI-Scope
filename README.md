# Embedded Scope

A 3D printed embedded AI-based microscope for pathology diagnosis.

### Requirements

The setup of this device requires two small single-board computers (ideally one of which should be specialized for deep learning, such as an Nvidia Jetson Nano). 
Current STLs support an Nvidia Jetson Nano for model inference and a Raspberry Pi for imaging processing. A Raspberry Pi Camera V2 is used for imaging of samples. The STLs support a battery pack at most as large as the Jetson Nano and a 7 inch touch screen, though the specific dimensions of the STLs could be used to accommodate for different setups (for example a larger screen, larger battery packs, or more than one battery). See ... for details on modifying the STLs.

### Instructions

STL files for 3D printing are in `/stls`. The prints have been tested using a stereolithography (SLA) 3D printer. STLs were generated in openscad and scripts can be found in `/stl_gen`. The scripts can be modified to accommodate for different part dimensions (especially nonspecific parts like battery pack(s) and touch screen).

To build the device,

* Please see [here](stls/assembly_instructions.pdf) for a list of parts and specific assembly instructions with pictures. 
* Also see [assembly video](https://drive.google.com/file/d/1WPFa4IFCZg4AjeARS-ab-TACosYXbkmb/view?usp=sharing) for a very general overview and timelapse of assembly. 
* Ethernet file transfer (if needed) between the Jetson Nano and the Raspberry Pi can be set up in many ways, such as using a combination of `inotifywait`, `watch`, and `scp`. See ... for details.
* After assembling and setting up the Ethernet connection (if needed), please see [model instructions](docs/README.md) for installing requirements and a toy example.

High-level overview of device structure and setup:

![Overview of device](docs/figs/overview.jpeg)

Comparison with traditional development and deployment setups:

![Pipeline flow](docs/figs/flow.jpeg)



