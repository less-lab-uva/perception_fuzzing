# Setting up Cityscapes 
Although the approach presented in the paper is applicable to many perception datasets, we target the [Cityscapes dataset](https://www.cityscapes-dataset.com/) in this implementation. 
The dataset is available for non-commercial use directly from Cityscapes.
The `CITYSCAPES_DATA_ROOT` must contain the `gtFine_trainvaltest` folder downloaded from Cityscapes.
Below is the example file tree.

```
CITYSCAPES_DATA_ROOT/gtFine_trainvaltest/
└── gtFine
    ├── leftImg8bit
    │   ├── test
    │   ├── train
    │   └── val
    ├── license.txt
    ├── README
    ├── test
    │   ├── berlin
    │   ├── bielefeld
    │   ├── bonn
    │   ├── leverkusen
    │   ├── mainz
    │   └── munich
    ├── train
    │   ├── aachen
    │   ├── bochum
    │   ├── bremen
    │   ├── cologne
    │   ├── darmstadt
    │   ├── dusseldorf
    │   ├── erfurt
    │   ├── hamburg
    │   ├── hanover
    │   ├── jena
    │   ├── krefeld
    │   ├── monchengladbach
    │   ├── strasbourg
    │   ├── stuttgart
    │   ├── tubingen
    │   ├── ulm
    │   ├── weimar
    │   └── zurich
    └── val
        ├── frankfurt
        ├── lindau
        └── munster

```

## Blurred Images
It is recommended that when presenting the images for humans, the blurred versions of the images are used to protect individual privacy. 
The [CityscapesMutator](src/images/image_mutator.py#L513) takes a boolean flag in the constructor noting if it should operate on the blurred dataset. 
If this flag is set, the `CITYSCAPES_DATA_ROOT` must contain the `leftImg8bit_blurred` folder downloaded from Cityscapes.
Below is the example file tree.
```
CITYSCAPES_DATA_ROOT/leftImg8bit_blurred/
├── leftImg8bit_blurred
│   ├── demoVideo
│   │   ├── stuttgart_00
│   │   ├── stuttgart_01
│   │   └── stuttgart_02
│   ├── test
│   │   ├── berlin
│   │   ├── bielefeld
│   │   ├── bonn
│   │   ├── leverkusen
│   │   ├── mainz
│   │   └── munich
│   ├── train
│   │   ├── aachen
│   │   ├── bochum
│   │   ├── bremen
│   │   ├── cologne
│   │   ├── darmstadt
│   │   ├── dusseldorf
│   │   ├── erfurt
│   │   ├── hamburg
│   │   ├── hanover
│   │   ├── jena
│   │   ├── krefeld
│   │   ├── monchengladbach
│   │   ├── strasbourg
│   │   ├── stuttgart
│   │   ├── tubingen
│   │   ├── ulm
│   │   ├── weimar
│   │   └── zurich
│   ├── train_extra
│   │   ├── augsburg
│   │   ├── bad-honnef
│   │   ├── bamberg
│   │   ├── bayreuth
│   │   ├── dortmund
│   │   ├── dresden
│   │   ├── duisburg
│   │   ├── erlangen
│   │   ├── freiburg
│   │   ├── heidelberg
│   │   ├── heilbronn
│   │   ├── karlsruhe
│   │   ├── konigswinter
│   │   ├── konstanz
│   │   ├── mannheim
│   │   ├── muhlheim-ruhr
│   │   ├── nuremberg
│   │   ├── oberhausen
│   │   ├── saarbrucken
│   │   ├── schweinfurt
│   │   ├── troisdorf
│   │   ├── wuppertal
│   │   └── wurzburg
│   └── val
│       ├── frankfurt
│       ├── lindau
│       └── munster
├── license.txt
└── README

```


## SUT results
The tool chain evaluates the systems under test (SUTs) on the original dataset.
This provides two functions.
First, as outlined in Section 6.2, tests that are too difficult for the SUTs do not warrant mutation as a mutation will provide no marginal benefit.
By running the best performing SUT based on its benchmark score on the dataset, we determine which test cases are too difficult and should be excluded.
Secondly, although the test cases generated by *semImFuzz* can be used to evaluate the SUTs on their own, we are particularly interested in the effect of the mutation itself. 
Thus, we compare the performance of the SUT on the original image to its performance on the mutated image. 

To facilitate efficient usage, the first time the tool chain is used, it will run the entire Cityscapes dataset on the SUTs and store their baseline performance on all of the original test cases. This will generate the `sut_gt_testing` folder, so named because it contains the SUTs' test performance for these *ground truth* test cases.

This folder will be automatically produced and have the following structure if using the five SUTs explored in the study:
```
CITYSCAPES_DATA_ROOT/sut_gt_testing/
├── decouple_segnet
├── decouple_segnet_raw_results.txt
├── efficientps
├── efficientps_raw_results.txt
├── hrnet
├── hrnet_raw_results.txt
├── mutations
├── mutations_gt
├── nvidia-sdcnet
├── nvidia-sdcnet_raw_results.txt
├── nvidia-semantic-segmentation
├── nvidia-semantic-segmentation_raw_results.txt
└── results.txt
```

For information on setting up the SUTs, please see [the study data README](study_data/README.md).