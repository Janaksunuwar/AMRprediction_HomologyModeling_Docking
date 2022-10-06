#Download the .pdb file of the subject for 2D MODELLER Alignment

blastOut_all1 = blastOut_all[['query', 'subject', 'qcov', 'pident']].reset_index(drop=True)
#remove the last two numbers from pdb accession (6 digits to 4 digits)
blastOut_all1[['pdb_sub', 'suffix' ]] = blastOut_all1['subject'].str.split('_', expand=True)
#remove the underline from query WP_001, as the underline seems to disrupt pir format (lets see...)
blastOut_all1['Nquery'] = blastOut_all1['query'].str.replace('_', '')
#final blastOut prepared
blastOut_align2d = blastOut_all1[['Nquery', 'pdb_sub', 'qcov', 'pident']]
#sort with higest qcov and pident
blastOut_align2d0 = blastOut_align2d.sort_values(['Nquery', 'qcov', 'pident'], ascending =(True, False, False))
#remove duplicates to keep the best hit for each query
blastOut_align2d1 = blastOut_align2d0.drop_duplicates(subset=['Nquery'], keep = 'first').reset_index(drop=True)

#download only the pdb files from RCSB databased from pdb IDs
pdb_dwnl_cmds = 'wget https://files.rcsb.org/download/' + blastOut_align2d1['pdb_sub'] + '.pdb'
for lines in pdb_dwnl_cmds:
    os.system(lines)
    
display(blastOut_align2d1)
