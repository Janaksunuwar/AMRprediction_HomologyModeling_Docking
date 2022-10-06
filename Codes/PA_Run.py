#All set, Intersection set and Random set performance for Pseudomonas aeruginosa

#Dictionary of antibiotics
antb_SN = {'imipenem': '3 a', 'meropenem': '3 b'}

#import pandas
import pandas as pd

for antb, SN in antb_SN.items():
    
    #Acronym for Pseudomonas aeruginosa
    bacteria = 'Pseudomonas'
    
    #Italicized full name bacteria for fig output
    italic_name = r"\textit{Pseudomonas aeruginosa}"
    
    #Figure plot number
    supplementary_fig_no = SN
    
    #Import gene-ast data from github repository
    file_name = f'https://github.com/Janaksunuwar/AMRprediction_HomologyModeling_Docking/raw/main/Data/Final_Gene_AST_matrix_{bacteria}_{antb}_qc70_pi30.csv'

    #no of validation
    validation_no = 6
    
    #Result out
    file_all_out = f"{bacteria}_{antb}_Complete_Results_{validation_no}-fold_CV.csv"
    
    #read the gene-ast data
    data_ = pd.read_csv(file_name)
    
    #Put all the susceptible class in a separate dataset
    susceptible = data_.loc[data_[antb] == 0]
    no_of= len(susceptible.index)
    
    #Select equal number of resistant dataset
    resistant = data_.loc[data_[antb] == 1].sample(n=no_of, random_state=42)
    
    #Concatenate both dataframes again
    data = pd.concat([resistant, susceptible])
    
    ML_Run()
