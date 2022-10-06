from sys import argv
from modeller import *

env = Environ()
aln = Alignment(env)
mdl = Model(env, file=argv[2], model_segment=('FIRST:' + argv[3],'LAST:' + argv[3]))
aln.append_model(mdl, align_codes=argv[2] + argv[3], atom_files=argv[2] + '.pdb')
aln.append(file=argv[1] + '.ali', align_codes=argv[1])
aln.align2d(max_gap_length=50)
aln.write(file=argv[1] + '-' + argv[2] + '.ali', alignment_format='PIR')
aln.write(file=argv[1]+ '-' + argv[2] + '.pap', alignment_format='PAP')
