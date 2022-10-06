#Homology Model Construction
#The best model for each protein is selected out of 5 models constructed based on highest MODELLER DOPE Score

total_models = len(blastOut_align2d1.index)
counter = 0
log_df = pd.DataFrame(columns=[])
for index, row in blastOut_align2d1.iterrows():
    query = row['Nquery']
    subject = row['pdb_sub']
    try:
        with open (f'{subject}.pdb', 'r') as f:
            for lines in f:
                if lines.startswith('COMPND   3 CHAIN:'):
                    chain = lines.split(': ')[1]
                    chain_a = chain[:1]
    except FileNotFoundError:
        print(f'{subject}.pdb file not found in PDB database')
        
    #run model-single.py
    cmd_modelsingle = f'mod10.1 model-single_Jan.py {query} {subject} {chain_a}'
    os.system(cmd_modelsingle)
    
    #open the out file and append the best  model to log_df
    with open ('model-single_Jan.log', 'r') as f:
        for lines in f:
            if lines.startswith('Top model:'):
                
                print(lines)
                log_df = log_df.append({'Best model': lines}, ignore_index=True)
