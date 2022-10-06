# AMRprediction_HomologyModeling_Docking
Using Machine learning framework to predict putative novel AMR genes and molecular docking studies to study interaction between respective antibiotics in various bacterial species.

The framework takes all the annotated genes and genomic features and AST data as labels to construct a binary matrix. The frameworks then prioritize the most important genes potentially responsible for respective antibiotic resistance by building models for All set, Intersection set, and Random set.

Further, an automated commandline molecular docking is performed by downloading RSCB protien database using MODELLER 10.1, custom command line align2d for template alignment based on dynamic programming algorithm, and five 3D models constructed by AutoMOdel class, then selection of the best model with highest DOPE score for docking each target template as described in Modeller tutorial.

The PDB models of receptors are prepared by removing heteroatoms, reparing hydorgens, and adding Kollman/Gasteiger charges as described in AutoDock. The SDF structure of the respective antiboitics are downloaded from PubChem and converted to PDB format with OpenBabel. The PDB ligands is prepared via command line Autodock ligand preparation. Then the homology modeled receptors (predicted protein) and ligands(antiboitics) binding free-energy affinity score (Kcal/mol) is calculated with AutoDock Vina Smina fork. 

# How to run
All the necessary files and .py files should be in the same directory and the terminal should indicate the same working directoy/path

Necessary Files: eg: Final_Gene_AST_matrix_Klebsiella_Doripenem_qc70_pi30.csv, which is a binary matrix of presence/absence genes for Klebsiella, antibiotic(Doripenem)

All the python .py files are in the Codes folder that can be downloaded to local working directory/can be user created

First of all ML_run.py file should be ran in local terminal as following. This will create a function Ml_run() that will create ML models for All set, Intersection Set and Random Set to produce figures of assessment of training, validation and test settings.

(base) @user$: python Ml_run.py 

Then, run the respective .py files of each bacteria to get  to get individual ML performance outputs of all three sets. These outputs would be named separately in the same directory.

(base) @user$: python Klebsiella_Run.py

