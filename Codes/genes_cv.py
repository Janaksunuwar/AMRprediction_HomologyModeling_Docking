#The set of consistent genes in all rounds of cross validation
import pandas as pd
import numpy as np
import os

def Table():
    All_HP_Lookup = pd.read_csv(f'{bact}_All_Hypothetical_Protein_Lookup.csv')
    table = pd.DataFrame(columns=[])
    for antb in antibios:
        #open the consistent genes
        cg_df = pd.read_csv(f'{bact}_{antb}_Consistent_Genes_Per_6-fold_CV.csv')
        #rename the column head Feature to match for merge
        cg_df = cg_df.rename(columns={'Feature': 'match'})
        #merge to keep only HP proteins from consistent genes and find accession number
        test = pd.merge(All_HP_Lookup, cg_df, on='match', how='inner', indicator=True)
        #remove the duplicate accession numbers
        test1 = test.drop_duplicates(subset=['non-redundant_refseq'])
        #have specific columns only
        consistent_lookedUp = test1[['assembly','non-redundant_refseq', 'name', 'match']].reset_index(drop=True)
        #Consistent genes for the publication table where all HP are replaced
        consistent_lookedUp_for_table = consistent_lookedUp.drop_duplicates(subset=['match']).reset_index(drop=True)
        #select only the name
        consistent_lookedUp_for_table_name_only = consistent_lookedUp_for_table[['non-redundant_refseq', 'name',]]
        d1 = consistent_lookedUp_for_table_name_only.rename(columns={'non-redundant_refseq': 'accession', 'name': f'{antb}'})
        table = pd.concat([table, d1], axis=1)
    #export to csv
    #table.to_csv(f'{bact}_consistent_genes_gene_level.csv')
    print(f'Putative consistent genes responsible for resistance in {bact}')
    display(table.head(10))

#For Klebsiella pneumoniae
bact = 'Klebsiella'
antibios = ['doripenem', 'ertapenem', 'imipenem', 'meropenem']   
Table()

#For E.coli and Shigella
bact = 'EcS'
antibios = ['doripenem', 'ertapenem', 'imipenem', 'meropenem']   
Table()

#For Pseudomonas aeruginosa
bact = 'Pseudomonas'
antibios = ['imipenem', 'meropenem']   
Table()

#For Enterobacter
bact = 'Enterobacter'
antibios = ['imipenem', 'meropenem']   
Table()

#For Salmonella enterica
bact = 'Salmonella'
antibios = ['gentamicin', 'kanamycin', 'streptomycin']   
Table()
