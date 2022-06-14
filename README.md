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
3. Set up the Python dependencies in [requirements.txt](requirements.txt)
```
pip3 install -r requirements.txt
```
4. Set up the environment variables in [init.bash](init.bash).
5. Download the [Cityscapes dataset](https://www.cityscapes-dataset.com/). See [this document](CityscapesSetup.md) for more information.
6. Each time you start a new shell to run the tester, run the following to set up the environment variables.
```
. init.bash
```
6. Set up the Systems Under Test (SUTs) that you intend to test with the system. For information on the five SUTs used in the study, please see the [study data README](study_data/README.md). This is not needed if you are only trying to recreate the mutated images from the study.

## Recreating the Tests from the Study
For information on how to recreate the tests from the study, please see [this README](study_data/README.md).