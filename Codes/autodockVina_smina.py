#Molecular Docking with AutoDock vina Smina fork
#The docking results can be visualized by AutoDock, PyMol, DiscoveryStudioVisualizer

for index, row in log_df.iterrows():
    receptor = row[1]
    receptor_name = row[1][:-3] + 'pdbqt'
        
    cmd_smina = f'smina -r {receptor_name} -l Doripenem.pdbqt --autobox_ligand {receptor_name} --cpu 7 --seed 1 --num_modes 5 --exhaustiveness 5 --energy_range 10 -o {receptor_name}_SMINA_OUT.pdbqt --log {receptor_name}_SMINA.log'
    os.system(cmd_smina)
