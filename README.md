# Embedded Scope

A 3D printed embedded AI-based microscope for pathology diagnosis.

STL files for 3D printing are in `/stls`. The prints have been tested using a stereolithography (SLA) 3D printer. STLs were generated in openscad and scripts can be found in `/stl_gen`. The scripts can be modified to accommodate for different part dimensions (especially nonspecific parts like battery pack(s) and touch screen).

To build the device, please see [parts list and instructions with images](stls/assembly_instructions.pdf) here. Also see [assembly video](https://drive.google.com/file/d/1WPFa4IFCZg4AjeARS-ab-TACosYXbkmb/view?usp=sharing) for a very general overview and timelapse of assembly. 

High-level overview of device structure and setup:

![Overview of device](docs/figs/overview.jpeg)

Comparison with traditional development and deployment setups:

![Pipeline flow](docs/figs/flow.jpeg)

After assembling and setting up the Ethernet connection (if needed), please see [model instructions](docs/README.md) for installing requirements and a toy example.

