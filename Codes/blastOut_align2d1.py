#For ALIGN 2D, MODELLER requires its default align2d.py 
# The modified file is provided in Github:https://github.com/Janaksunuwar/AMRprediction_HomologyModeling_Docking/tree/main/example_Homology_Modeling_Data

for index, row in blastOut_align2d1.iterrows():
    query = row['Nquery']
    subject = row['pdb_sub']
    #open and change CHAIN: A
    try:
        with open (f'{subject}.pdb', 'r') as f:
            for lines in f:
                if lines.startswith('COMPND   3 CHAIN:'):
                    chain = lines.split(': ')[1]
                    chain_a = chain[:1]
        cmd_PB = f'mod10.1 align2d_Jan.py {query} {subject} {chain_a}'
        os.system(cmd_PB)
    except:
        print(f'{subject}.pdb file not found in PDB database')
