## Student Project by maxschae

In the notebook *student_project.ipynb* the main results of the following paper are replicated: M. Brückner and A. Ciccone (2011) "Rain and the Democratic Window of Opportunity", *Econometrica*, Vol. 79, pp. 923-947. The paper can be found [here]( https://doi.org/10.3982/ECTA8183), the paper's data are available [here](https://www.econometricsociety.org/content/supplement-rain-and-democratic-window-opportunity-0), and additional data on elections and geographic information are taken from [here](https://www.idea.int/data-tools/continent-view/Africa/40) and [here](http://www.maplibrary.org/library/stacks/Africa/index.htm), respectively.

The authors find that non-persistent negative shocks to income raise institutional scores in sub-Saharan African countries in the following year - countries become less autocratic and more democratic. To overcome endogeneity and reverse causality issues an instrumental variable approach is taken, where rainfall serves as an instrument for GDP per capita.

While the main results of the paper will be replicated, additional work focuses around the

* visualisation of the underlying story, 
* the interpretation of results in light of the expositions on the *Local Average Treatment Effect* and *Average Causal Response*, and 
* an extension of the not-much discussed mechanism that relates temporary income shocks to improvements in institutional scores. For that, the role of presidential and parliamentary elections is investigated.

Please feel free to ask any question, or to comment on the work.


<hr />


The repository includes folders for data, graphs, tables and TeX-files. All regression outputs are readout as TeX-files and fixed effect coefficients are manually erased. The final file to produce the tables presented is stored in the same folder. Further, the TeX-file written to produce causal graphs is placed in the TeX-files folder. For reasons of reproducibility and transparency all functions producing regression outputs are called in the Appendix. Hence, the reader should not be afraid of the notebook's length since it emerges from the blown-up Appendix.

<br>

[![Build Status](https://travis-ci.org/HumanCapitalAnalysis/student-project-maxschae.svg?branch=master)](https://travis-ci.org/HumanCapitalAnalysis/student-project-maxschae) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/HumanCapitalAnalysis/student-project-maxschae/master?filepath=student_project.ipynb) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](HumanCapitalAnalysis/student-project-maxschae/blob/master/LICENSE)