#coding:gbk
from sklearn.metrics import f1_score,auc,precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit
import pickle
import sys
import timeit

import numpy as np
from sklearn.preprocessing import label_binarize
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.init import uniform_

from sklearn.metrics import auc,roc_auc_score,roc_curve, precision_score,precision_recall_curve, recall_score,f1_score
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "3"


class MiRNADiseaseAssociationPrediction(nn.Module):
    def __init__(self):
        super(MiRNADiseaseAssociationPrediction, self).__init__()
        self.embed_fingerprint = nn.Embedding(n_fingerprint+1, dim)
        self.bond_embed = nn.Linear(10, dim)
        self.neighbor_fc = nn.ModuleList([nn.Linear(dim*2, dim)
                                    for _ in range(layer_gnn)])
        self.side_fc = nn.ModuleList([nn.Linear(dim, dim)
                                          for _ in range(layer_gnn)])
        self.W_sub = nn.ModuleList([nn.Linear(dim, dim)
                                    for _ in range(radius)])

        self.fc = nn.Linear(n_seqFeature, dim)
        uniform_(self.fc.weight,-0.01,0.01)


        self.W_attention = nn.Linear(dim, dim)
        self.W_out = nn.ModuleList([nn.Linear(2*dim, 2*dim)
                                    for _ in range(layer_output)])
        self.W_interaction = nn.Linear(2*dim, n_classes)

    def sub_graph(self,xs, A):
        for i in range(radius):
            hs = torch.relu(self.W_sub[i](xs))
            xs = xs + torch.matmul(A, hs)
        return xs
    def gnn(self, xs, atom_list, bond_feature, bond_list, i_bond_j, layer):
        bond_list = bond_list.long()
        atom_list = atom_list.long()
        bond_feature = self.bond_embed(bond_feature)
        atom_f = xs
        for i in range(layer):
            bond_neighbor = bond_feature[bond_list]
            atom_neighbor = atom_f[atom_list]
            neighbor_feature = torch.cat([atom_neighbor,bond_neighbor],dim=-1)
            neighbor_feature = F.leaky_relu(self.neighbor_fc[i](neighbor_feature))
            atom_f = F.sigmoid(atom_f+torch.sum(neighbor_feature,dim=-2))#equation 5 and 6
            bond_feature = F.sigmoid(bond_feature + self.side_fc[i](torch.sum(atom_f[i_bond_j.long()],dim=-2)) )#equation 7 and 8
        return torch.unsqueeze(torch.mean(xs+atom_f, 0), 0)


    def forward(self, inputs):

        fingerprints, atom_degree_list, bond_feature, bond_degree_list, i_bond_j, adjacency, words = inputs

        """disease vector with GNN."""
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        fingerprint_vectors = self.sub_graph(fingerprint_vectors, adjacency)
        disease_vector = self.gnn(fingerprint_vectors, atom_degree_list, bond_feature, bond_degree_list, i_bond_j, layer_gnn)

        """miRNA vector with MLP."""
        words = words.view(1,-1)
        miRNA_vector = self.fc(words)

        """Concatenate the above two vectors and output the interaction."""
        cat_vector = torch.cat((disease_vector, miRNA_vector), 1)
        for j in range(layer_output):
            cat_vector = torch.relu(self.W_out[j](cat_vector))
        association = self.W_interaction(cat_vector)

        return association

    def __call__(self, data, train=True):

        inputs, correct_association = data[:-1], data[-1]

        predicted_association = self.forward(inputs)

        if train:
            loss = F.cross_entropy(predicted_association, correct_association)
            return loss
        else:
            correct_labels = correct_association.to('cpu').data.numpy()
            ys = F.softmax(predicted_association, 1).to('cpu').data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = ys
            return correct_labels, predicted_labels, predicted_scores


class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=lr, weight_decay=weight_decay)

    # start added by zhao
    def train(self, dataset, mirnafeaturemat, diseasefeaturemat):
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        for data_i, datain in enumerate(dataset):
            #### one_mirna_feature
            # print(data_i)
            one_mirnaid, one_diseaseid, interaction = datain[1], datain[0], datain[-1]
            mirna_feature = mirnafeaturemat[int(one_mirnaid), :]
            mirna_feature = np.array(list(mirna_feature), dtype=np.float32)

            #### one_disease_feature
            finger, atom_degree_list, bond_feature, bond_degree_list, i_bond_j, adjacency_new = diseasefeaturemat[
                int(one_diseaseid)]

            atom_degree_list = torch.from_numpy(atom_degree_list).to(device)
            bond_feature = torch.from_numpy(bond_feature).to(device)
            bond_degree_list = torch.from_numpy(bond_degree_list).to(device)
            i_bond_j = torch.from_numpy(i_bond_j).to(device)
            finger_tensor = torch.from_numpy(finger).to(device)
            adjacency_tensor = torch.from_numpy(adjacency_new).to(device).float()
            mirna_feature_tensor = torch.from_numpy(mirna_feature).to(device)
            datafeaturein = (
                finger_tensor, atom_degree_list, bond_feature, bond_degree_list, i_bond_j, adjacency_tensor,
                mirna_feature_tensor, interaction)


            loss = self.model(datafeaturein)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.to('cpu').data.numpy()
        return loss_total

class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataset, mirnafeaturemat, diseasefeaturemat):
        N = len(dataset)
        T, Y, S = [], [], []

        for datain in dataset:
            #### one_mirna_feature
            one_mirnaid, one_diseaseid, interaction = datain[1], datain[0], datain[-1]
            mirna_feature = mirnafeaturemat[int(one_mirnaid), :]
            mirna_feature = np.array(list(mirna_feature))

            #### one_disease_feature
            finger, atom_degree_list, bond_feature, bond_degree_list, i_bond_j, adjacency_new = diseasefeaturemat[
                int(one_diseaseid)]

            atom_degree_list = torch.from_numpy(atom_degree_list).to(device)
            bond_feature = torch.from_numpy(bond_feature).to(device)
            bond_degree_list = torch.from_numpy(bond_degree_list).to(device)
            i_bond_j = torch.from_numpy(i_bond_j).to(device)
            finger_tensor = torch.from_numpy(finger).to(device)
            adjacency_tensor = torch.from_numpy(adjacency_new).to(device).float()
            mirna_feature_tensor = torch.from_numpy(mirna_feature).to(device)
            datafeaturein = (
                finger_tensor, atom_degree_list, bond_feature, bond_degree_list, i_bond_j, adjacency_tensor,
                mirna_feature_tensor, interaction)
            # end zqc

            (correct_labels, predicted_labels,
                 predicted_scores) = self.model(datafeaturein, train=False)
            T.append(correct_labels)
            Y.append(predicted_labels)
            S.append(predicted_scores)

        y_one_hot = label_binarize(T, np.arange(n_classes))
        S = np.array(S)
        S = np.reshape(S, (-1, n_classes))
        AUC = roc_auc_score(y_one_hot, S, average='macro')
        AUC_per = per(y_one_hot, Y, S)

        return AUC,  AUC_per, y_one_hot, Y, S

    def save_result(self, result, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, result)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)


def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy',allow_pickle=True)]


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def load_strlist(file_name):
    readlist = []
    with open(file_name, 'r') as f:
        for line in f:
            readlist.append(line.strip())
    f.close()
    return readlist

def load_matrix(file_name):
    with open(file_name, "r") as inf:
        matrix = [line.strip("\n").split()[0:] for line in inf]
    inf.close()
    matrix = np.array(matrix, dtype=np.float32)
    return matrix

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2

def per(y_one_hot,Y, y_pred):
    Y=label_binarize(Y, np.arange(n_classes))


    AUC=np.zeros(n_classes)

    for i in range(n_classes):
        AUC[i] = roc_auc_score(y_one_hot[:,i], y_pred[:,i])
    return AUC



if __name__ == "__main__":

    DATASET = 'miRNAdisease'

    # radius=1
    radius = 2
    # radius=3

    # ngram=2
    ngram = 3

    dim = 10
    layer_gnn = 3
    side = 5
    window =(2 * side + 1)
    layer_cnn = 3
    layer_output = 3
    lr = 1e-3
    lr_decay = 0.5
    decay_interval = 25
    weight_decay = 1e-6
    iteration = 100
    setting = 'test'

    AUC = []
    n_classes=6
    AUC_per_MLP = np.zeros(n_classes)

    """CPU or GPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    """Load preprocessed data."""
    dir_input = ('./data/miRNAdisease-HMDD2.0-task3/')

    CV_mirnaid_list = load_strlist(dir_input + 'CVmiRNA_all_and_un_ass.txt')
    CV_diseaseid_list = load_strlist(dir_input + 'CVdisease_all_and_un_ass.txt')
    associations = load_tensor(dir_input + 'label_all_and_un_ass', torch.LongTensor)

    mirnafeaturemat = load_matrix(dir_input + 'HMDD2_new_miRNAseqfeature.txt')
    n_seqFeature = len(mirnafeaturemat[0,:])

    diseasefeaturemat = np.load(dir_input + "diseasefeature-r.npy", allow_pickle=True)

    fingerprint_dict = load_pickle(dir_input + 'fingerprint_dict.pickle')
    n_fingerprint = len(fingerprint_dict)

    print(len(associations))

    """Create a dataset and split it into train/dev/test."""
    dataset = list(zip(CV_diseaseid_list, CV_mirnaid_list, associations))

    fold_time=0
    dataset = shuffle_dataset(dataset, 23)

    for kk in range(1):
        """Output files."""
        file_output = './output/PDMDA-task3-result.txt'
        Results = ('Epoch\tTime(sec)\tLoss_train\tAUC_test')
        with open(file_output, 'w') as f:
            f.write(Results + '\n')

        dataset=np.array(dataset)
        skf = StratifiedShuffleSplit(n_splits=5,train_size=0.8,test_size=0.2)

        for train_index, test_index in skf.split(dataset, dataset[:,-1]):
            fold_time=fold_time+1

            print(fold_time)
            dataset_train, dataset_test = dataset[train_index], dataset[test_index]

            """Set a model."""
            torch.manual_seed(1234)
            model = MiRNADiseaseAssociationPrediction().to(device)
            trainer = Trainer(model)
            tester = Tester(model)

            """Start training."""
            print('Training...')
            print(Results)
            start = timeit.default_timer()

            for epoch in range(1, iteration):
                if epoch % decay_interval == 0:
                    trainer.optimizer.param_groups[0]['lr'] *= lr_decay
                loss_train = trainer.train(dataset_train,mirnafeaturemat,diseasefeaturemat)

                AUC_test, AUC_per, TT, YY, SS = \
                      tester.test(dataset_test,mirnafeaturemat,diseasefeaturemat)

                end = timeit.default_timer()
                time = end - start

                Results = [epoch, time, loss_train, AUC_test]
                tester.save_result(Results, file_output)
                print('\t'.join(map(str, Results)))
                if epoch == iteration - 1:
                    AUC.append(AUC_test)
                    AUC_per_MLP = np.concatenate((AUC_per_MLP, AUC_per), axis=0)


    AUC_per_MLP=np.reshape(AUC_per_MLP,(-1,n_classes))
    AUC_per_MLP = np.delete(AUC_per_MLP,0,axis = 0)


    Results = ['Mean AUC:',np.mean(AUC)]
    tester.save_result(Results, file_output)
    print(np.mean(AUC))

    Results = ['AUC_per_type:',np.mean(AUC_per_MLP,axis=0)]
    tester.save_result(Results, file_output)
    print('AUC_per_type:', np.mean(AUC_per_MLP, axis=0))