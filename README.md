
=======================================================================================================
Welcome to use PDMDA algorithm. PDMDA is a novel and effective computational model for predicting deep-level miRNA-disease association.
-------------------------------------------------------------------------------------------------------

## Requirements
PyTorch 1.1.0

###		Author:Cheng Yan and Jianxin Wang
###		E-mail:jxwang@mail.csu.edu.cn yancheng01@mail.csu.edu.cn
=======================================================================================================
PDMDA_task1.py: the 5-fold cross validation of predicting association type between known 4 association type samples
PDMDA_task2.py: the 5-fold cross validation of predicting type between known 4 association type samples and un-type samples
PDMDA_task3.py: the 5-fold cross validation of predicting the type between known 4 association type samples, untype samples and un-ass samples



###data/miRNAdisease-HMDD2.0-task1 directory
CVdisease_types.txt:the diseases id of dataset (known 4 association type samples)
CVmiRNA_types.txt:the miRNAs id of dataset (known 4 association type samples)
label_types.npy: the association type labels of miRNA-disease pairs of dataset (known 4 association type samples)
HMDD2_new_miRNAseqfeature.txt: the features of miRNAs
diseasefeature-r.zip: the raduis feature of disease (unzip it to diseasefeature-r.npy)
disease_list.txt: the uniquely diseases id of dataset
fingerprint_dict.pickle: the uniquely gene dict of dataset
disease_gene.pickle: the disease-gene interactions
geneintmat.zip: the gene-gene interaction (unzip it to geneintmat.txt)



###data/miRNAdisease-HMDD2.0-task2 directory
CVdisease_all_and_un_types.txt:the diseases id of dataset (known 4 association type samples and un-type samples)
CVmiRNA_all_and_un_types.txt:the miRNAs id of dataset (known 4 association type samples and un-type samples)
label_all_and_un_types.npy: the association type labels of miRNA-disease pairs of dataset (known 4 association type samples and un-type samples)
HMDD2_new_miRNAseqfeature.txt: the features of miRNAs
diseasefeature-r.zip: the raduis feature of disease (unzip it to diseasefeature-r.npy)
disease_list.txt: the uniquely diseases id of dataset
fingerprint_dict.pickle: the uniquely gene dict of dataset
disease_gene.pickle: the disease-gene interactions
geneintmat.zip: the gene-gene interaction (unzip it to geneintmat.txt)



###data/miRNAdisease-HMDD2.0-task3 directory
CVdisease_all_and_un_ass.txt:the diseases id of dataset (known 4 association type samples, untype samples and un-ass samples)
CVmiRNA_all_and_un_ass.txt:the miRNAs id of dataset (known 4 association type samples, untype samples and un-ass samples)
label_all_and_un_ass.npy: the association type labels of miRNA-disease pairs of dataset (known 4 association type samples, untype samples and un-ass samples)
HMDD2_new_miRNAseqfeature.txt: the features of miRNAs
diseasefeature-r.zip: the raduis feature of disease (unzip it to diseasefeature-r.npy)
disease_list.txt: the uniquely diseases id of dataset
fingerprint_dict.pickle: the uniquely gene dict of dataset
disease_gene.pickle: the disease-gene interactions
geneintmat.zip: the gene-gene interaction (unzip it to geneintmat.txt)