# False Positive Report
This folder contains the details of the false positive evaluation conducted for Research Question 1 in the paper.
As described in Section 5.4.1, in order to understand the false positivity rate of *semImFuzz*, we manually inspected all of the mutations that led to a 5+ p.p.
drop and then sampled 10% of the test cases that produced a drop between 1 and 5 p.p.
Each of the three authors then voted on whether the resulting mutation was a true positive or false positive.
If any of the authors voted false positive, it was marked as such.
These findings are recorded in [false_positive.txt](false_positive.txt) and [true_positive.txt](true_positive.txt) respectively.
These files contain the names of the mutations generated based on the corresponding mutation parameters used in the study.
Please see [this README](../README.md) for information on how to generate these mutations.


The results reported in the paper and shown below list the false positivity rate as a percentage of the test cases that resulted in drops of that size for the given SUT.
Since the same mutation may have resulted in different outcomes for the different SUTs, the determinations of false positive vs true positive given here must be cross-referenced with the individual SUT performance to obtain these results.


<table>
<tr><th rowspan="2"><b>SUT</b></th><th colspan="3">False Positive Rate</th></tr>
<tr><td>[1, 5)</td><td>[5, 10)</td><td>[10, 100]</td></tr>
<tr><td>NVIDIA SemSeg</td><td>11%</td><td>67%</td><td>50%</td></tr>
<tr><td>EfficientPS</td><td>43%</td><td>47%</td><td>53%</td></tr>
<tr><td>DecoupleSegNet</td><td>38%</td><td>75%</td><td>-</td></tr>
<tr><td>SDCNet</td><td>42%</td><td>0%</td><td>0%</td></tr>
<tr><td>HRNetV2+OCR</td><td>28%</td><td>67%</td><td>-</td></tr>
</table>
Table 2: False Positive Rate for Inconsistencies Found