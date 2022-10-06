#Fasta to Pir Modeller format conversion
path1 = os.getcwd()
fnames2=[]
for files in os.listdir(path1):
    if files.endswith("_.fasta"):
        fnames2.append(files)

for i in range(0, len(fnames2)):
    file = fnames2[i]
    f_na = file.split('.fasta')[0]
    accn = f_na.replace('_', '')

    with open(file, "r") as f, open (accn + '.ali', "w") as output:
        sequence=""
        for line in f:
            if line.startswith(">"):
                output.write(f'>P1;{accn}' + '\n')
                output.write(f'sequence:{accn}:::::::0.00: 0.00' + '\n')
            else:
                line=''.join(line.split()) 
                output.write(line)
            if not line:
                line = line.rstrip()
        output.write('*')
