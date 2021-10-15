from modeller import *
from modeller.automodel import *
import sys
from sys import argv

class MyModel(AutoModel):
    def get_model_filename(self, root_name, id1, id2, file_ext):
        return "%s_%s_model_%d%s" % (self.knowns[0], root_name, id2, file_ext)


chain = argv[2] + argv[3]
out_file_name = argv[1] + argv[2]


#from modeller import soap_protein_od
log.verbose()
env = Environ()
a = MyModel(env, alnfile=argv[1]+'-'+ argv[2]+'.ali',
              knowns= chain, sequence=argv[1],
              assess_methods=(assess.DOPE,
                              #soap_protein_od.Scorer(),
                              assess.GA341))
a.starting_model = 1 
a.ending_model = 5 
a.make()



# Get a list of all successfully built models from a.outputs
ok_models = [x for x in a.outputs if x['failure'] is None]
#ok_models = [x for x in a.outputs if x['failure'] is None]
# Rank the models by DOPE score
key = 'DOPE score'
if sys.version_info[:2] == (2,3):
    # Python 2.3's sort doesn't have a 'key' argument
    ok_models.sort(lambda a,b: cmp(a[key], b[key]))
else:
    ok_models.sort(key=lambda a: a[key])

# Get top model
m = ok_models[0]
print("Top model: %s (DOPE score %.3f)" % (m['name'], m[key]))
print("Query vs Subject: %s" % (out_file_name))
print('Good to go JanBro New')

#os.rename('%s'%(m['name']), '%s.pdb' %(out_file_name))
#with open ('test_best_homology_model', 'a') as f:
#    #file.write("yaaaa JanBro")
#    f.write(out)

