from Bio import SeqIO
import numpy as np
import random
import scipy
from Models_Evolution import Models_Evolution
class Engine_Evolution():
    def __init__(self, sequence, time, intensity_matrix, print_datas=True):
        self.sequence = sequence
        self.probability_matrix = scipy.linalg.expm(intensity_matrix * time)
        self.nucleotids_to_matrix = {'A': 0, 'G': 1, 'C': 2, 'T': 3}
        if print_datas:
            print('Q-matrix: ')
            print(intensity_matrix)
            print(f'P-matrix: ')
            print(self.probability_matrix)

    def get_new_sequence(self, path_to_save=None, sequence_description='>'):
        if path_to_save:
            new_sequence = sequence_description + '\n'
        else:
            new_sequence = ''
        for nucleotide in self.sequence:
            nucleotide_place_in_probability_matrix = self.nucleotids_to_matrix[nucleotide]
            probability_list_for_this_nucleotid = self.probability_matrix[nucleotide_place_in_probability_matrix]
            new_nucleotid = random.choices(['A', 'G', 'C', 'T'], weights=probability_list_for_this_nucleotid)[0]
            new_sequence += new_nucleotid
        if path_to_save:
            f = open(path_to_save, 'w')
            f.write(new_sequence)
            f.close()
        return new_sequence
if __name__ == "__main__":
    path = input('Enter path to fasta file or enter "RANDOM" for generate random sequence: ')
    choice = input('Choice a model of nucleotide substitutions:\n1) JC69 (Jukes and Cantor 1969)\n2) K81 (Kimura 1981)\n3) F81 (Felsenstein 1981)\n4) GTR (Tavare 1986)\n5) Random model with random params\n==>')
    me = Models_Evolution()
    if choice == '1':
        random_or_no = input('Select the desired:\nGenerate random params for JC69 — 1\nEnter the parameters yourself — 2 ')
        if random_or_no == '1':
            me.generate_random_intensity_matrix(rand_value=0)
        else:
            alpha = float(input('Enter alpha: '))
            intensity_matrix = me.JC69(alpha)
    elif choice == '2':
        random_or_no = input('Select the desired:\nGenerate random params for K81 — 1\nEnter the parameters yourself — 2 ')
        if random_or_no == '1':
            me.generate_random_intensity_matrix(rand_value=1)
        else:
            alpha = float(input('Enter alpha: '))
            beta = float(input('Enter beta: '))
            gamma = float(input('Enter gamma: '))
            intensity_matrix = me.K81(alpha, beta, gamma)
    elif choice == '3':
        random_or_no = input('Select the desired:\nGenerate random params for F81 — 1\nEnter the parameters yourself — 2 ')
        if random_or_no == '1':
            me.generate_random_intensity_matrix(rand_value=2)
        else:
            frequency_A = float(input('Enter adenin frequency: '))
            frequency_G = float(input('Enter guanine frequency: '))
            frequency_C = float(input('Enter cytosine frequency: '))
            frequency_T = float(input('Enter thymine frequency: '))
            intensity_matrix = me.F81(frequency_A, frequency_G, frequency_C, frequency_T)
    elif choice == '4':
        random_or_no = input('Select the desired:\nGenerate random params for GTR — 1\nEnter the parameters yourself — 2 ')
        if random_or_no == '1':
            me.generate_random_intensity_matrix(rand_value=3)
        else:
            alpha = float(input('Enter alpha (A_to_G): '))
            beta = float(input('Enter beta (A_to_C): '))
            lamda = float(input('Enter lambda (A_to_T): '))
            delta = float(input('Enter delta (G_to_C): '))
            epsilon = float(input('Enter epsilon (G_to_T): '))
            eta = float(input('Enter eta (C_to_T): '))
            frequency_A = float(input('Enter adenin frequency: '))
            frequency_G = float(input('Enter guanine frequency: '))
            frequency_C = float(input('Enter cytosine frequency: '))
            frequency_T = float(input('Enter thymine frequency: '))
            intensity_matrix = me.GTR(alpha, beta, gamma, lamda, delta, epsilon, eta, frequency_A, frequency_G, frequency_C, frequency_T)
    elif choice == '5':
        intensity_matrix = me.generate_random_intensity_matrix()[1]
    if path == "RANDOM":
        sequence = ''.join(random.choice('ATCG') for i in range(10**6))
        sequence_description = '>'
    else:
        for record in SeqIO.parse(path, "fasta"):
            sequence_description = '>' + record.description
            sequence = record.seq
    time = float(input('Enter time: '))
    path_to_save = input('Enter path to save new generated sequence or enter "PRINT" for only print datas: ')
    engine = Engine_Evolution(sequence, time, intensity_matrix)
    if path_to_save == 'PRINT':
        print(engine.get_new_sequence())
    else:
        engine.get_new_sequence(path_to_save = path_to_save, sequence_description=sequence_description)
