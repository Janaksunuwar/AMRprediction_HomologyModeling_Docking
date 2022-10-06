#Ligand download and ligand preparation
ligand_name=['Doripenem']
#Download ligand
for Abs in ligand_name:
        cmd_dwnld_pubchem = f'wget https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{Abs}/SDF?record_type=3d -O {Abs}.sdf'
        os.system(cmd_dwnld_pubchem)
#path to MGL tools prepare_ligand.py
prepare_ligand4_py = '/home/js1349/Desktop/AutoDOCK/mgltools_x86_64Linux2_1.5.6/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py'

#Using openbabel to convert SDF file to PDB
cmd_sdf_to_pdb = 'obabel -isdf Doripenem.sdf -opdb > Doripenem.pdb'
os.system(cmd_sdf_to_pdb)

#Ligand prep
'''
l- ligand file
-A repairs types-bonds_hydrogens, bonds, hydrogens
-o pdbqt format output
'''
cmd_prepare_ligand4= f'{pythonsh} {prepare_ligand4_py} -l Doripenem.pdb -A hydrogens -o Doripenem.pdbqt'
os.system(cmd_prepare_ligand4)
print('Ligand Prepared')
