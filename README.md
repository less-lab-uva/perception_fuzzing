# Semantic Image Fuzzing of AI Perception Systems
This repository contains code and high level algorithm descriptions for the paper 
[Semantic Image Fuzzing of AI Perception Systems (ICSE '22)](Semantic%20Image%20Fuzzing%20of%20AI%20Perception%20Systems.pdf).


## Setting up the codebase
1. Clone this repository
2. Initialize and update the git submodules in the root of the repository. This sets up the [Cityscapes scripts dependency](https://github.com/less-lab-uva/cityscapesScripts). This can be done by:
```
git submodule init
git submodule update
```
3. Set up the environment variables in [init.bash](init.bash).
4. Download the [Cityscapes dataset](https://www.cityscapes-dataset.com/). See [this document](CityscapesSetup.md) for more information.
5. Each time you start a new shell to run the tester, run the following to set up the environment variables.
```
. init.bash
```
6. Set up the Systems Under Test (SUTs) that you intend to test with the system. For information on the five SUTs used in the study, please see the [study data README](study_data/README.md). This is not needed if you are only trying to recreate the mutated images from the study.

## Recreating the Tests from the Study
For information on how to recreate the tests from the study, please see [this README](study_data/README.md).


## Repository Status
The repository is currently undergoing edits to improve readability and reproducability. 
This page will be updated as these additions are made. The starting point for the code is the 
[tester.py](src/images/tester.py) script. There are currently several hard-coded file paths 
that need to be manually changed to run the script locally - these will be made configurable soon.

These updates will include:
* High level algorithm descriptions in Markdown format linking to the algorithms described in [image_mutator.py](/src/images/image_mutator.py).
* A guide for downloading and setting up the repository and its dependencies (e.g. Cityscapes) along with refactoring to add parameters for all file paths.
* A top level script for reproducing the images used in the paper's study.
* Additional in-line comments in the code to ease understanding and reproducability.