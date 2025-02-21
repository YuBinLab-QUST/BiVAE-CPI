# -*- coding: utf-8 -*
import os  
import pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from collections import defaultdict


atom_dict = defaultdict(lambda: len(atom_dict))
bond_dict = defaultdict(lambda: len(bond_dict))
fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
edge_dict = defaultdict(lambda: len(edge_dict))
word_dict = defaultdict(lambda: len(word_dict))


def create_atoms(mol):
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], 'aromatic')
    atoms = [atom_dict[a] for a in atoms]
    return np.array(atoms)


def create_ijbonddict(mol):
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))

    atoms_set = set(range(mol.GetNumAtoms()))
    isolate_atoms = atoms_set - set(i_jbond_dict.keys())
    bond = bond_dict['nan']
    for a in isolate_atoms:
        i_jbond_dict[a].append((a, bond))

    return i_jbond_dict


def atom_features(atoms, i_jbond_dict, radius):
    if (len(atoms) == 1) or (radius == 0):
        fingerprints = [fingerprint_dict[a] for a in atoms]

    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict
        for _ in range(radius):
            fingerprints = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                fingerprints.append(fingerprint_dict[fingerprint])

            nodes = fingerprints
            _i_jedge_dict = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge = edge_dict[(both_side, edge)]
                    _i_jedge_dict[i].append((j, edge))
            i_jedge_dict = _i_jedge_dict

    return np.array(fingerprints)


def create_adjacency(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    adjacency = np.array(adjacency)
    adjacency += np.eye(adjacency.shape[0], dtype=int)
    return adjacency


def get_fingerprints(mol):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024, useChirality=True)
    return fp.ToBitString()


def split_sequence(sequence, ngram):
    sequence = '-' + sequence + '='
    words = [word_dict[sequence[i:i + ngram]]
             for i in range(len(sequence) - ngram + 1)]
    return np.array(words)

def dump_dictionary(dictionary, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dict(dictionary), f)


def get_matrix(data_list):
    compound_dict = {}
    protein_dict = {}
    c = 0
    p = 0
    for no, data in enumerate(data_list):
        smiles, sequence, _ = data.strip().split(" ")
        if smiles not in compound_dict.keys():
            compound_dict[smiles] = c
            c += 1
        if sequence not in protein_dict.keys():
            protein_dict[sequence] = p
            p += 1

    compound_nums = len(compound_dict)
    protein_nums = len(protein_dict)
    data_matrix = np.zeros((compound_nums, protein_nums))
    print(data_matrix.shape)
    for no, data in enumerate(data_list):
        smiles, sequence, interaction = data.strip().split(" ")
        if interaction == '1':
            c_idx = compound_dict[smiles]
            p_idx = protein_dict[sequence]
            data_matrix[c_idx, p_idx] = 1
    return data_matrix, compound_dict, protein_dict

def extract_input_data(input_path, output_path, radius, ngram):
    with open(input_path, "r") as f:
        data_list = f.read().strip().split('\n')

    data_list = [d for d in data_list if '.' not in d.strip().split()[0]]
    N = len(data_list)
    data_matrix, compound_map, protein_map = get_matrix(data_list)

    compounds, adjacencies, fps, proteins, interactions = [], [], [], [], []
    compound_dict, protein_dict = {}, {}
    for no, data in enumerate(data_list):
        smiles, sequence, interaction = data.strip().split(" ")

        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        atoms = create_atoms(mol)
        i_jbond_dict = create_ijbonddict(mol)

        compounds.append(atom_features(atoms, i_jbond_dict, radius))
        adjacencies.append(create_adjacency(mol))
        fps.append(get_fingerprints(mol))
        proteins.append(split_sequence(sequence, ngram))
        interactions.append(np.array([float(interaction)]))
        compound_dict[no] = compound_map.get(smiles)
        protein_dict[no] = protein_map.get(sequence)

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    np.save(os.path.join(output_path, 'compounds'), compounds)
    np.save(os.path.join(output_path, 'adjacencies'), adjacencies)
    np.save(os.path.join(output_path, 'fingerprint'), fps)
    np.save(os.path.join(output_path, 'proteins'), proteins)
    np.save(os.path.join(output_path, 'interactions'), interactions)
    np.save(os.path.join(output_path, 'data_matrix'), data_matrix)
    dump_dictionary(compound_dict, os.path.join(output_path, 'compound_dict'))
    dump_dictionary(protein_dict, os.path.join(output_path, 'protein_dict'))



if __name__=='__main__':
    dataset = 'human'
    input_path = '../data/' + dataset + '/' + '3/' + 'data.txt'
    output_path = '../dataset/' + dataset + '/3'
    radius, ngram = 2, 3
    extract_input_data(input_path, output_path, radius, ngram)

    dump_dictionary(fingerprint_dict, os.path.join(output_path, 'atom_dict'))
    dump_dictionary(word_dict, os.path.join(output_path, 'amino_dict'))
    print('save successfully')


