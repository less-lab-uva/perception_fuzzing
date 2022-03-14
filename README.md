# Semantic Image Fuzzing of AI Perception Systems
This repository contains code and high level algorithm descriptions for the paper 
Semantic Image Fuzzing of AI Perception Systems (ICSE '22).

The repository is currently undergoing edits to improve readability and reproducability. 
This page will be updated as these additions are made. The starting point for the code is the 
[tester.py](src/images/tester.py) script. There are currently several hard-coded file paths 
that need to be manually changed to run the script locally - these will be made configurable soon.

These updates will include:
* High level algorithm descriptions in Markdown format linking to the algorithms described in [image_mutator.py](/src/images/image_mutator.py).
* A guide for downloading and setting up the repository and its dependencies (e.g. Cityscapes) along with refactoring to add parameters for all file paths.
* A top level script for reproducing the images used in the paper's study.
* Additional in-line comments in the code to ease understanding and reproducability.