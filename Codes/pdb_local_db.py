#download PDB sequences in FASTA format, make blast database locally
import os
import pandas as pd

cmd_pdb = 'wget https://ftp.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt.gz'
os.system(cmd_pdb)

#unzip pdb sequence
cmd_unz = 'gunzip *.gz'
os.system(cmd_unz)

#make blast db
cmd_blst = "makeblastdb -in pdb_seqres.txt -dbtype prot -out PDB_protDb"
os.system(cmd_blst)

#concate query sequences
cmd_q_cat = 'cat *_.fasta > query_all.faa'
os.system(cmd_q_cat)

#blastp
cmd14= f"blastp -query query_all.faa -db PDB_protDb -out blastp_out -evalue 1E-5 -outfmt '6 qseqid sseqid qcovs pident evalue' -max_target_seqs 50000"
os.system(cmd14)

#make blastout a dataframe
blastOut = pd.read_csv("blastp_out", sep="\t", names=["query", "subject", "qcov", "pident", "Evalue"])

#remove "gb|" made by blast in subject
blastOut['subject'] = blastOut['subject'].str.replace("gb|", '')
blastOut['subject'] = blastOut['subject'].str.replace("|", '')

#select query and subject beyond a threshold
blastOut_all = blastOut.loc[(blastOut['qcov'] >= 70) & (blastOut['pident'] >= 30)]

display(blastOut_all.head(15))
