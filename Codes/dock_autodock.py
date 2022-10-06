#Molecular docking with AutoDock
#Receptor preparation
"""
-r  receptor_file
-o  outfile.pdbqt
-A  add hydorgen
-U  waters cleanup, default on, no flag required
-C  do not add charges, default on, no flag required
"""    
#The path to pythonsh is required by AutoDock tools
pythonsh='/home/js1349/Desktop/AutoDOCK/mgltools_x86_64Linux2_1.5.6/bin/pythonsh'
#Prepare_receptor4.py file required by AutoDock tools
prepare_receptor_py = '/home/js1349/Desktop/AutoDOCK/mgltools_x86_64Linux2_1.5.6/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py'
#test_cmd = f'{pythonsh} prepare_receptor4.py -r {pdb_file} -o jan4try -A checkhydrogens'
log_df['BM names'] = log_df['Best model'].str.split(' ').str[2]
for index, row in log_df.iterrows():
    receptor = row[1]
    receptor_name = row[1][:-3]
    cmd_prepare_receptor4 = f'{pythonsh} {prepare_receptor_py} -r {receptor} -U waters -A hydrogens -o {receptor_name}pdbqt'
    os.system(cmd_prepare_receptor4 )
