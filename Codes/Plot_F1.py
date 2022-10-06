#Plot Figure 2, Figure 3, Figure 4, and Figure 5

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib import rc

#Activate latex text rendering
rc('text', usetex=True)

#Italic bacterial names for figures
it1 = r"\textit{Klebsiella pneumoniae}"
it2 = r"\textit{E. coli}"
it2_ = r"\textit{Shigella}"
it3 = r"\textit{Pseudomonas aeruginosa}"
it4 = r"\textit{Enterobacter}"
it5 = r"\textit{Salmonella enterica}"

# vlidation number
val_no = 6

#Open files for figure 2
d1 = pd.read_csv(f'Klebsiella_doripenem_F1_comparision_{val_no}-fold_CV.csv')
d1.set_index(['classifier'], inplace=True)
d1_stdev = pd.read_csv(f'Klebsiella_doripenem_F1_stdev_comparision_{val_no}-fold_CV.csv')
yerr_d1 = d1_stdev.iloc[:, 1:].to_numpy().T

d2 = pd.read_csv(f'EcS_doripenem_F1_comparision_{val_no}-fold_CV.csv')
d2.set_index(['classifier'], inplace=True)
d2_stdev = pd.read_csv(f'EcS_doripenem_F1_stdev_comparision_{val_no}-fold_CV.csv')
yerr_d2 = d2_stdev.iloc[:, 1:].to_numpy().T

d3 = pd.read_csv(f'Pseudomonas_imipenem_F1_comparision_{val_no}-fold_CV.csv')
d3.set_index(['classifier'], inplace=True)
d3_stdev = pd.read_csv(f'Pseudomonas_imipenem_F1_stdev_comparision_{val_no}-fold_CV.csv')
yerr_d3 = d3_stdev.iloc[:, 1:].to_numpy().T

#Bar colors and set name
#my_colors=['khaki', 'red', 'skyblue']
my_labels = ['All Set', 'Intersection Set', 'Random Set']

#Subplots for figure 2
fig, axes = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=False, figsize=(16,5))
plt.xlabel("")

ax1 = d1.plot(kind='bar', width = 0.8, ax=axes[0], yerr= yerr_d1, error_kw=dict(lw = 0.5, capsize = 1, capthick =0.5), 
         title= f"2a. Accuracy F1 on {it1} for Doripenem",
         legend=False, yticks=np.arange(0,1.1, step=0.1),
         fontsize=8, rot=0)
ax1.set_xlabel('')
ax1.set_axisbelow(True)

ax2 = d2.plot(kind='bar', width = 0.8, ax=axes[1], yerr= yerr_d2, error_kw=dict(lw = 0.5, capsize = 1, capthick =0.5), 
         title= f"2b. Accuracy F1 on {it2} and {it2_} for Doripenem",
         legend=False, yticks=np.arange(0,1.1, step=0.1),
         fontsize=8, rot=0)
ax2.set_xlabel('')
ax2.set_axisbelow(True)

ax2.legend(labels=my_labels, loc='lower center', borderaxespad=0.1, ncol=3,
           bbox_to_anchor=(0.5, -0.25),
           fancybox=False, shadow=False, prop={'size': 8})


ax3 = d3.plot(kind='bar', width = 0.8, ax=axes[2], yerr= yerr_d3, error_kw=dict(lw = 0.5, capsize = 1, capthick =0.5), 
         title= f"2c. Accuracy F1 on {it3} for Imipenem",
         legend=False, yticks=np.arange(0,1.1, step=0.1),
         fontsize=8, rot=0)
ax2.set_xlabel('')
ax2.set_axisbelow(True)

fig.tight_layout()
plt.show()

fig.savefig(f'Figure2_gene_level.tiff', dpi=300, format="tiff", pil_kwargs={"compression": "tiff_lzw"})

#Open files for figure 3
d1j = pd.read_csv(f'Klebsiella_meropenem_F1_comparision_{val_no}-fold_CV.csv')
d1j.set_index(['classifier'], inplace=True)
d1j_stdev = pd.read_csv(f'Klebsiella_meropenem_F1_stdev_comparision_{val_no}-fold_CV.csv')
yerr_d1j = d1j_stdev.iloc[:, 1:].to_numpy().T

d2j = pd.read_csv(f'EcS_meropenem_F1_comparision_{val_no}-fold_CV.csv')
d2j.set_index(['classifier'], inplace=True)
d2j_stdev = pd.read_csv(f'EcS_meropenem_F1_stdev_comparision_{val_no}-fold_CV.csv')
yerr_d2j = d2j_stdev.iloc[:, 1:].to_numpy().T

d3j = pd.read_csv(f'Enterobacter_meropenem_F1_comparision_{val_no}-fold_CV.csv')
d3j.set_index(['classifier'], inplace=True)
d3j_stdev = pd.read_csv(f'Enterobacter_meropenem_F1_stdev_comparision_{val_no}-fold_CV.csv')
yerr_d3j = d3j_stdev.iloc[:, 1:].to_numpy().T

#Lable names for legend
my_labels = ['All Set', 'Intersection Set', 'Random Set']

#Subplots for figure 3
fig, axes = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=False, figsize=(16,5))
plt.xlabel("")

ax1 = d1j.plot(kind='bar', width = 0.8, ax=axes[0], yerr= yerr_d1j, error_kw=dict(lw = 0.5, capsize = 1, capthick =0.5), 
         title= f"3a. Accuracy F1 on {it1} for Meropenem",
         legend=False, yticks=np.arange(0,1.1, step=0.1),
         fontsize=8, rot=0)
ax1.set_xlabel('')
ax1.set_axisbelow(True)

ax2 = d2j.plot(kind='bar', width = 0.8, ax=axes[1], yerr= yerr_d2j, error_kw=dict(lw = 0.5, capsize = 1, capthick =0.5), 
         title= f"3b. Accuracy F1 on {it1} and {it2_} for Meropenem",
         legend=False, yticks=np.arange(0,1.1, step=0.1),
         fontsize=8, rot=0)
ax2.set_xlabel('')
ax2.set_axisbelow(True)

ax2.legend(labels=my_labels, loc='lower center', borderaxespad=0.1, ncol=3,
           bbox_to_anchor=(0.5, -0.25),
           fancybox=False, shadow=False, prop={'size': 8})
ax2.set_axisbelow(True)

ax3 = d3j.plot(kind='bar', width = 0.8, ax=axes[2], yerr= yerr_d3j, error_kw=dict(lw = 0.5, capsize = 1, capthick =0.5), 
         title= f"3c. Accuracy F1 on {it4} for Meropenem",
         legend=False, yticks=np.arange(0,1.1, step=0.1),
         fontsize=8, rot=0)
ax3.set_xlabel('')
ax3.set_axisbelow(True)

fig.tight_layout()
plt.show()

fig.savefig(f'Figure3_gene_level.tiff', dpi=300, format="tiff", pil_kwargs={"compression": "tiff_lzw"})

#Open files for figure 4
d4 = pd.read_csv(f'Klebsiella_ertapenem_F1_comparision_{val_no}-fold_CV.csv')
d4.set_index(['classifier'], inplace=True)
d4_stdev = pd.read_csv(f'Klebsiella_ertapenem_F1_stdev_comparision_{val_no}-fold_CV.csv')
yerr_d4 = d4_stdev.iloc[:, 1:].to_numpy().T

d5 = pd.read_csv(f'Klebsiella_imipenem_F1_comparision_{val_no}-fold_CV.csv')
d5.set_index(['classifier'], inplace=True)
d5_stdev = pd.read_csv(f'Klebsiella_imipenem_F1_stdev_comparision_{val_no}-fold_CV.csv')
yerr_d5 = d5_stdev.iloc[:, 1:].to_numpy().T

d6 = pd.read_csv(f'EcS_ertapenem_F1_comparision_{val_no}-fold_CV.csv')
d6.set_index(['classifier'], inplace=True)
d6_stdev = pd.read_csv(f'EcS_ertapenem_F1_stdev_comparision_{val_no}-fold_CV.csv')
yerr_d6 = d6_stdev.iloc[:, 1:].to_numpy().T

d7 = pd.read_csv(f'EcS_imipenem_F1_comparision_{val_no}-fold_CV.csv')
d7.set_index(['classifier'], inplace=True)
d7_stdev = pd.read_csv(f'EcS_imipenem_F1_stdev_comparision_{val_no}-fold_CV.csv')
yerr_d7 = d7_stdev.iloc[:, 1:].to_numpy().T

d8 = pd.read_csv(f'Pseudomonas_meropenem_F1_comparision_{val_no}-fold_CV.csv')
d8.set_index(['classifier'], inplace=True)
d8_stdev = pd.read_csv(f'Pseudomonas_meropenem_F1_stdev_comparision_{val_no}-fold_CV.csv')
yerr_d8 = d8_stdev.iloc[:, 1:].to_numpy().T

d9 = pd.read_csv(f'Enterobacter_imipenem_F1_comparision_{val_no}-fold_CV.csv')
d9.set_index(['classifier'], inplace=True)
d9_stdev = pd.read_csv(f'Enterobacter_imipenem_F1_stdev_comparision_{val_no}-fold_CV.csv')
yerr_d9 = d9_stdev.iloc[:, 1:].to_numpy().T

#Subplots for figure 4
fig, axes = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=True, figsize=(16,12))
plt.xlabel("")

ax4 = d4.plot(kind='bar', width = 0.8, ax=axes[0,0], yerr= yerr_d4, error_kw=dict(lw = 0.5, capsize = 1, capthick =0.5), 
         title= f"4a. Accuracy F1 on {it1} for Ertapenem",
         legend=False, yticks=np.arange(0,1.1, step=0.1),
         fontsize=8, rot=0)
ax4.set_xlabel('')
ax4.set_axisbelow(True)

ax5 = d5.plot(kind='bar', width = 0.8, ax=axes[0,1], yerr= yerr_d5, error_kw=dict(lw = 0.5, capsize = 1, capthick =0.5), 
         title= f"4b. Accuracy F1 on {it1} for Imipenem",
         legend=False, yticks=np.arange(0,1.1, step=0.1),
         fontsize=8, rot=0)
ax5.set_xlabel('')
ax5.set_axisbelow(True)

ax6 = d6.plot(kind='bar', width = 0.8, ax=axes[1,0], yerr= yerr_d6, error_kw=dict(lw = 0.5, capsize = 1, capthick =0.5), 
         title= f"4c. Accuracy F1 on {it2} and {it2_} for Ertapenem",
         legend=False, yticks=np.arange(0,1.1, step=0.1),
         fontsize=8, rot=0)
ax6.set_xlabel('')
ax6.set_axisbelow(True)


ax7 = d7.plot(kind='bar', width = 0.8, ax=axes[1,1], yerr= yerr_d7, error_kw=dict(lw = 0.5, capsize = 1, capthick =0.5), 
         title= f"4d. Accuracy F1 on {it2} and {it2_} for Imipenem",
         legend=False, yticks=np.arange(0,1.1, step=0.1),
         fontsize=8, rot=0)
ax7.set_xlabel('')
ax7.set_axisbelow(True)

ax8 = d8.plot(kind='bar', width = 0.8, ax=axes[2,0], yerr= yerr_d8, error_kw=dict(lw = 0.5, capsize = 1, capthick =0.5), 
         title= f"4e. Accuracy F1 on {it3} for Meropenem",
         legend=False, yticks=np.arange(0,1.1, step=0.1),
         fontsize=8, rot=0)
ax8.set_xlabel('')
ax8.set_axisbelow(True)

ax8.legend(labels=my_labels, loc='lower center', borderaxespad=0.1, ncol=3,
           bbox_to_anchor=(0.5, -0.29),
           fancybox=False, shadow=False, prop={'size': 11})

ax9 = d9.plot(kind='bar', width = 0.8, ax=axes[2,1], yerr= yerr_d9, error_kw=dict(lw = 0.5, capsize = 1, capthick =0.5), 
         title= f"4f. Accuracy F1 on {it4} for Imipenem",
         legend=False, yticks=np.arange(0,1.1, step=0.1),
         fontsize=8, rot=0)
ax9.set_xlabel('')
ax9.set_axisbelow(True)

ax9.legend(labels=my_labels, loc='lower center', borderaxespad=0.1, ncol=3,
           bbox_to_anchor=(0.5, -0.27),
           fancybox=False, shadow=False, prop={'size': 11})

plt.subplots_adjust(top = 0.97, bottom=0.0, hspace=0.29, wspace=0.05 )
fig.tight_layout(pad=1.5)
plt.show()

fig.savefig(f'Fig4_gene_level.tiff', dpi=300, format="tiff", pil_kwargs={"compression": "tiff_lzw"})

#Open files for figure 5
d10 = pd.read_csv(f'Salmonella_gentamicin_F1_comparision_{val_no}-fold_CV.csv')
d10.set_index(['classifier'], inplace=True)
d10_stdev = pd.read_csv(f'Salmonella_gentamicin_F1_stdev_comparision_{val_no}-fold_CV.csv')
yerr_d10 = d10_stdev.iloc[:, 1:].to_numpy().T

d11 = pd.read_csv(f'Salmonella_kanamycin_F1_comparision_{val_no}-fold_CV.csv')
d11.set_index(['classifier'], inplace=True)
d11_stdev = pd.read_csv(f'Salmonella_kanamycin_F1_stdev_comparision_{val_no}-fold_CV.csv')
yerr_d11 = d11_stdev.iloc[:, 1:].to_numpy().T

d12 = pd.read_csv(f'Salmonella_streptomycin_F1_comparision_{val_no}-fold_CV.csv')
d12.set_index(['classifier'], inplace=True)
d12_stdev = pd.read_csv(f'Salmonella_streptomycin_F1_stdev_comparision_{val_no}-fold_CV.csv')
yerr_d12 = d12_stdev.iloc[:, 1:].to_numpy().T

#Subplots for figure 5
fig, axes = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=False, figsize=(16,5))
plt.xlabel("")

ax10 = d10.plot(kind='bar', width = 0.8, ax=axes[0], yerr= yerr_d10, error_kw=dict(lw = 0.5, capsize = 1, capthick =0.5), 
         title= f"5a. Accuracy F1 on {it5} for Gentamicin",
         legend=False, yticks=np.arange(0,1.1, step=0.1),
         fontsize=8, rot=0)
ax10.set_xlabel('')
ax10.set_axisbelow(True)
ax10.margins(0)

ax11 = d11.plot(kind='bar', width = 0.8, ax=axes[1], yerr= yerr_d11, error_kw=dict(lw = 0.5, capsize = 1, capthick =0.5), 
         title= f"5b. Accuracy F1 on {it5} for Kanamycin",
         legend=False, yticks=np.arange(0,1.1, step=0.1),
         fontsize=8, rot=0)
ax11.set_xlabel('')
ax11.set_axisbelow(True)

ax11.legend(labels=my_labels, loc='lower center', borderaxespad=0.1, ncol=3,
           bbox_to_anchor=(0.5, -0.2),
           fancybox=False, shadow=False, prop={'size': 11})
ax11.set_axisbelow(True)

ax12 = d12.plot(kind='bar', width = 0.8, ax=axes[2], yerr= yerr_d12, error_kw=dict(lw = 0.5, capsize = 1, capthick =0.5), 
         title= f"5c. Accuracy F1 on {it5} for Streptomycin",
         legend=False, yticks=np.arange(0,1.1, step=0.1),
         fontsize=8, rot=0)
ax12.set_xlabel('')
ax12.set_axisbelow(True)

fig.tight_layout()
plt.show()

fig.savefig(f'Fig5_gene_level.tiff', dpi=300, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
