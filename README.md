# AMRprediction_HomologyModeling_Docking
Using Machine learning framework to predict putative novel AMR genes and molecular docking studies to study interaction between antibiotics

The framework takes all the annotated genes and genomic features and AST data as labels to construct a binary matrix. The frameworks then prioritize the most important genes potentially responsible for respective antibiotic resistance by building models for All set, Intersection set, and Random set.

Further, an automated commandline molecular docking is performed by downloading RSCB protien database using MODELLER 10.1, custom command line align2d for template alignment based on dynamic programming algorithm, and five 3D models constructed by AutoMOdel class, then selection of the best model with highest DOPE score for docking each target template as described in Modeller tutorial.


