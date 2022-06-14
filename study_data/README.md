# Recreating the Study Data
This folder contains the information needed to recreate the study data from Section 5 in the paper.
## Recreating the Mutations
For the study, we performed three types of mutation: changing the color of a car, adding a car, and adding a pedestrian. 
The mutations were carried out in batches of 6000 with 2000 each of the different mutations.
The `all_mutation_logs` folder contains 25 files, each with 6000 mutation parameters that describe each of the mutations produced.
Together, these 25 files provide the full parameters needed to generate the 150,000 mutations from the study.
The [recreate_mutations.py](../src/images/recreate_mutations.py) script contains code to automatically re-generate all of the images from the study.
The script will run out-of-the-box to re-create the mutations, but cannot be used for running the SUTs on the images.
See below for more information on running the SUTs.

## Running the SUTs
To study the utility of *semImFuzz* to test perception systems, we set up the five best performing systems on the [Cityscapes Benchmark](https://www.cityscapes-dataset.com/benchmarks/#scene-labeling-task) that provided code and model data at the time of the study. The leaderboard is constantly changing, and there may be newer SUTs available.

To facilitate recreation, we forked the five repositories and updated the code so that it could be containerized in Docker and tested. Each of the repositories linked below has its own README with instructions to download the model weights and build the docker container.
1. NVIDIASemSeg: https://github.com/less-lab-uva/semantic-segmentation/tree/7726b14
2. EfficientPS: https://github.com/less-lab-uva/EfficientPS
3. DecoupleSegNets: https://github.com/less-lab-uva/DecoupleSegNets
4. SDCNet: https://github.com/less-lab-uva/semantic-segmentation/tree/sdcnet
5. HRNetV2+OCR: https://github.com/less-lab-uva/HRNet-Semantic-Segmentation

## Examining False Positivity Rates
Information on the false positivity study carried out for the paper can be found in the [false_positive_report](false_positive_report) folder.
The false positivity information is expressed using the names of the mutations generated in the study, so it is recommended to first recreate the mutations using the method above. 


## Recreating Timing Data
Research Question 2 explored the efficiency of the mutation process in terms of the amount of time it took to generate the test case compared to the amount of time it took the SUT to run on the test case for each of the different types of mutations.
The study used [this Python script](../src/images/run_timing_tests.py) to generate or run 100 tests 10 times.
The tests were run on a pop-os 18.04 desktop with an Intel Xeon Silver
4216 processor, 128GB of RAM, and an NVIDIA TITAN RTX GPU with 24GB of
VRAM.
These results are reported in Table 3 in Section 5.4.2, reproduced below.
<table>
<tr><th><b>Activity</b></th><th><b>Add Car</b></th><th><b>Add Person</b></th><th><b>Color Car</b></th></tr>
<tr><td><b>Test Generation</b></td><td><b>593 (19.9)</b></td><td><b>642 (42.7)</b></td><td><b>105 (3.06)</b></td></tr>
<tr><td>NVIDIA SemSeg</td><td>3421 (29.5)</td><td>3418 (22.9)</td><td>3404 (31.5)</td></tr>
<tr><td>EfficientPS</td><td>837 (7.22)</td><td>837 (5.88)</td><td>837 (7.41)</td></tr>
<tr><td>DecoupleSegNet</td><td>1426 (2.24)</td><td>1425 (2.60)</td><td>1423 (2.48)</td></tr>
<tr><td>SDCNet</td><td>1328 (3.09)</td><td>1328 (2.81)</td><td>1327 (8.33)</td></tr>
<tr><td>HRNetV2+OCR</td><td>600 (5.89)</td><td>605 (4.23)</td><td>605 (2.89)</td></tr>
</table>
Table 3: Average time to generate and execute a test in milliseconds, with the standard deviation in parentheses.