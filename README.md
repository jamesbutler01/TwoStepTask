# Neural data and analysis code for Miranda, Butler et al. (2025)



This repository contains all neural data, and code to generate the figures, from the study:



`Miranda Bruno, Butler James L, Malalasekera W M Nishantha, Behrens Timothy EJ, Dayan Peter, Kennerley Steven W (2025) Neural signatures of model-based and model-free reinforcement learning across prefrontal cortex and striatum eLife 14:RP106032`



This includes the raw spike timestamps and task event timings for 240 ACC neurons, 187 DLPFC neurons, 115 Caudate neurons, and 119 Putamen neurons from two subjects playing our modified version of the Daw Two-Step task. 



To get started simply run one of the `Fig.` scripts to generate the relevant figure. The first time running will be considerably slower as the smoothed spike rasters for each cell and trial epoch are generated from the raw data. Subsequent runs will be quicker as these rasters are cached locally upon creation. Note, the code for statistics/permutations is not provided due to the extensive time these take to run, but we're happy to provide them upon request. 



Please do cite our paper if you decide to use the data. If you have any questions then feel free to get in touch with us via the corresponding author information listed on the Elife paper. 

