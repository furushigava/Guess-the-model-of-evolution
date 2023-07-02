import numpy as np
from Bio import SeqIO
import scipy
import math
from Models_Evolution import Models_Evolution
class Model_Evolution_Predictor():
    def __init__(self, sequence_last, sequence_now, time):
        self.sequence_last = sequence_last
        self.sequence_now = sequence_now
        self.time = time
        self.nucleotids_to_matrix = {'A': 0, 'G': 1, 'C': 2, 'T': 3}
        self.models_of_evolution_class = Models_Evolution()
        if len(sequence_now) != len(sequence_last):
            return 'Error! The length does not match!'
    def calc_emperical_matrix(self):
        emperical_probability_matrix = np.array([[0, 0, 0, 0],
                                     [0, 0, 0, 0],
                                     [0, 0, 0, 0],
                                     [0, 0, 0, 0]])
        self.to_divisions = [0,0,0,0]
        i = 0
        while i < len(self.sequence_now):
            nucleotid_now = self.sequence_now[i]
            nucleotid_last = self.sequence_last[i]
            nucleotid_now_place_in_matrix = self.nucleotids_to_matrix[nucleotid_now]
            nucleotid_last_place_in_matrix = self.nucleotids_to_matrix[nucleotid_last]
            emperical_probability_matrix[nucleotid_last_place_in_matrix][nucleotid_now_place_in_matrix] += 1
            self.to_divisions[nucleotid_last_place_in_matrix] += 1
            i += 1
        emperical_probability_matrix = emperical_probability_matrix / self.to_divisions
        emperical_intensity_matrix = (scipy.linalg.logm(emperical_probability_matrix)) / self.time
        #emperical_intensity_matrix = np.real(emperical_intensity_matrix)
        return emperical_intensity_matrix, emperical_probability_matrix

    def calc_delta_matrix_JC69(self, alpha):
        matrix_by_alpha = self.models_of_evolution_class.JC69(alpha)
        delta_arr = (self.emperical_intensity_matrix - matrix_by_alpha) ** 2
        return delta_arr.sum()
    def calc_delta_matrix_K81(self, alpha, beta, gamma):
        matrix_by_alpha = self.models_of_evolution_class.K81(alpha, beta, gamma)
        delta_arr = (self.emperical_intensity_matrix - matrix_by_alpha) ** 2
        return delta_arr.sum()
    def calc_delta_matrix_F81(self, frequency_A, frequency_G, frequency_C, frequency_T):
        matrix_by_alpha = self.models_of_evolution_class.F81(frequency_A, frequency_G, frequency_C, frequency_T)
        delta_arr = (self.emperical_intensity_matrix - matrix_by_alpha) ** 2
        return delta_arr.sum()
    def calc_delta_matrix_GTR(self, A_to_G, A_to_C, A_to_T, G_to_C, G_to_T, C_to_T, frequency_A, frequency_G, frequency_C, frequency_T):
        matrix_by_alpha = self.models_of_evolution_class.GTR(A_to_G, A_to_C, A_to_T, G_to_C, G_to_T, C_to_T, frequency_A, frequency_G, frequency_C, frequency_T)
        delta_arr = (self.emperical_intensity_matrix - matrix_by_alpha) ** 2
        return delta_arr.sum()

    def calc_derivative_for_one_param_JC69(self, alpha, check_param = 'alpha', dx=0.0001):
        return (self.calc_delta_matrix_JC69(alpha + dx) - self.calc_delta_matrix_JC69(alpha)) / dx
    def calc_derivative_for_one_param_K81(self, alpha, beta, gamma, check_param, dx=0.0001):
        if check_param == 'alpha':
            return (self.calc_delta_matrix_K81(alpha + dx, beta, gamma) - self.calc_delta_matrix_K81(alpha, beta, gamma)) / dx
        elif check_param == 'beta':
            return (self.calc_delta_matrix_K81(alpha, beta + dx, gamma) - self.calc_delta_matrix_K81(alpha, beta, gamma)) / dx
        elif check_param == 'gamma':
            return (self.calc_delta_matrix_K81(alpha, beta, gamma + dx) - self.calc_delta_matrix_K81(alpha, beta, gamma)) / dx
    def calc_derivative_for_one_param_F81(self, frequency_A, frequency_G, frequency_C, frequency_T, check_param, dx=0.0001):
        if check_param == 'frequency_A':
            return (self.calc_delta_matrix_F81(frequency_A + dx, frequency_G, frequency_C, frequency_T) - self.calc_delta_matrix_F81(frequency_A, frequency_G, frequency_C, frequency_T)) / dx
        elif check_param == 'frequency_G':
            return (self.calc_delta_matrix_F81(frequency_A, frequency_G + dx, frequency_C, frequency_T) - self.calc_delta_matrix_F81(frequency_A, frequency_G, frequency_C, frequency_T)) / dx
        elif check_param == 'frequency_C':
            return (self.calc_delta_matrix_F81(frequency_A, frequency_G, frequency_C + dx, frequency_T) - self.calc_delta_matrix_F81(frequency_A, frequency_G, frequency_C, frequency_T)) / dx
        elif check_param == 'frequency_T':
            return (self.calc_delta_matrix_F81(frequency_A, frequency_G, frequency_C, frequency_T + dx) - self.calc_delta_matrix_F81(frequency_A, frequency_G, frequency_C, frequency_T)) / dx
    def calc_derivative_for_one_param_GTR(self, A_to_G, A_to_C, A_to_T, G_to_C, G_to_T, C_to_T, frequency_A, frequency_G, frequency_C, frequency_T, check_param, dx=0.0001):
        if check_param == 'A_to_G':
            return (self.calc_delta_matrix_GTR(A_to_G + dx, A_to_C, A_to_T, G_to_C, G_to_T, C_to_T, frequency_A, frequency_G, frequency_C, frequency_T) - self.calc_delta_matrix_GTR(A_to_G, A_to_C, A_to_T, G_to_C, G_to_T, C_to_T, frequency_A, frequency_G, frequency_C, frequency_T)) / dx
        elif check_param == 'A_to_C':
            return (self.calc_delta_matrix_GTR(A_to_G, A_to_C + dx, A_to_T, G_to_C, G_to_T, C_to_T, frequency_A, frequency_G, frequency_C, frequency_T) - self.calc_delta_matrix_GTR(A_to_G, A_to_C, A_to_T, G_to_C, G_to_T, C_to_T, frequency_A, frequency_G, frequency_C, frequency_T)) / dx
        elif check_param == 'A_to_T':
            return (self.calc_delta_matrix_GTR(A_to_G, A_to_C, A_to_T + dx, G_to_C, G_to_T, C_to_T, frequency_A, frequency_G, frequency_C, frequency_T) - self.calc_delta_matrix_GTR(A_to_G, A_to_C, A_to_T, G_to_C, G_to_T, C_to_T, frequency_A, frequency_G, frequency_C, frequency_T)) / dx
        elif check_param == 'G_to_C':
            return (self.calc_delta_matrix_GTR(A_to_G, A_to_C, A_to_T, G_to_C + dx, G_to_T, C_to_T, frequency_A, frequency_G, frequency_C, frequency_T) - self.calc_delta_matrix_GTR(A_to_G, A_to_C, A_to_T, G_to_C, G_to_T, C_to_T, frequency_A, frequency_G, frequency_C, frequency_T)) / dx
        elif check_param == 'G_to_T':
            return (self.calc_delta_matrix_GTR(A_to_G, A_to_C, A_to_T, G_to_C, G_to_T + dx, C_to_T, frequency_A, frequency_G, frequency_C, frequency_T) - self.calc_delta_matrix_GTR(A_to_G, A_to_C, A_to_T, G_to_C, G_to_T, C_to_T, frequency_A, frequency_G, frequency_C, frequency_T)) / dx
        elif check_param == 'C_to_T':
            return (self.calc_delta_matrix_GTR(A_to_G, A_to_C, A_to_T, G_to_C, G_to_T, C_to_T + dx, frequency_A, frequency_G, frequency_C, frequency_T) - self.calc_delta_matrix_GTR(A_to_G, A_to_C, A_to_T, G_to_C, G_to_T, C_to_T, frequency_A, frequency_G, frequency_C, frequency_T)) / dx
        elif check_param == 'frequency_A':
            return (self.calc_delta_matrix_GTR(A_to_G, A_to_C, A_to_T, G_to_C, G_to_T, C_to_T, frequency_A + dx, frequency_G, frequency_C, frequency_T) - self.calc_delta_matrix_GTR(A_to_G, A_to_C, A_to_T, G_to_C, G_to_T, C_to_T, frequency_A, frequency_G, frequency_C, frequency_T)) / dx
        elif check_param == 'frequency_G':
            return (self.calc_delta_matrix_GTR(A_to_G, A_to_C, A_to_T, G_to_C, G_to_T, C_to_T, frequency_A, frequency_G + dx, frequency_C, frequency_T) - self.calc_delta_matrix_GTR(A_to_G, A_to_C, A_to_T, G_to_C, G_to_T, C_to_T, frequency_A, frequency_G, frequency_C, frequency_T)) / dx
        elif check_param == 'frequency_C':
            return (self.calc_delta_matrix_GTR(A_to_G, A_to_C, A_to_T, G_to_C, G_to_T, C_to_T, frequency_A, frequency_G, frequency_C + dx, frequency_T) - self.calc_delta_matrix_GTR(A_to_G, A_to_C, A_to_T, G_to_C, G_to_T, C_to_T, frequency_A, frequency_G, frequency_C, frequency_T)) / dx
        elif check_param == 'frequency_T':
            return (self.calc_delta_matrix_GTR(A_to_G, A_to_C, A_to_T, G_to_C, G_to_T, C_to_T, frequency_A, frequency_G, frequency_C, frequency_T + dx) - self.calc_delta_matrix_GTR(A_to_G, A_to_C, A_to_T, G_to_C, G_to_T, C_to_T, frequency_A, frequency_G, frequency_C, frequency_T)) / dx
    def calc_best_params_for_JC69(self, step=0.01, count_of_epochs=1000):
        alpha = 100
        for _ in range(count_of_epochs):
            alpha -= step * self.calc_derivative_for_one_param_JC69(alpha, check_param = 'alpha')
        result_delta = self.calc_delta_matrix_JC69(alpha)
        return [[alpha], result_delta]
    def calc_best_params_for_K81(self, step=0.01, count_of_epochs=1000):
        alpha = 100
        beta = 100
        gamma = 100
        for _ in range(count_of_epochs):
            alpha -= step * self.calc_derivative_for_one_param_K81(alpha, beta, gamma, check_param = 'alpha')
            beta  -= step * self.calc_derivative_for_one_param_K81(alpha, beta, gamma, check_param = 'beta')
            gamma -= step * self.calc_derivative_for_one_param_K81(alpha, beta, gamma, check_param = 'gamma')
        result_delta = self.calc_delta_matrix_K81(alpha, beta, gamma)
        return [[alpha, beta, gamma], result_delta]
    def calc_best_params_for_F81(self, step=0.01, count_of_epochs=1000):
        frequency_A = 100
        frequency_G = 100
        frequency_C = 100
        frequency_T = 100
        for _ in range(count_of_epochs):
            frequency_A -= step * self.calc_derivative_for_one_param_F81(frequency_A, frequency_G, frequency_C, frequency_T, check_param = 'frequency_A')
            frequency_G -= step * self.calc_derivative_for_one_param_F81(frequency_A, frequency_G, frequency_C, frequency_T, check_param = 'frequency_G')
            frequency_C -= step * self.calc_derivative_for_one_param_F81(frequency_A, frequency_G, frequency_C, frequency_T, check_param = 'frequency_C')
            frequency_T -= step * self.calc_derivative_for_one_param_F81(frequency_A, frequency_G, frequency_C, frequency_T, check_param = 'frequency_T')
        result_delta = self.calc_delta_matrix_F81(frequency_A, frequency_G, frequency_C, frequency_T)
        return [[frequency_A, frequency_G, frequency_C, frequency_T], result_delta]

    def calc_best_params_for_GTR(self, step=0.01, count_of_epochs=1000):
        A_to_G = 100
        A_to_C = 100
        A_to_T = 100
        G_to_C = 100
        G_to_T = 100
        C_to_T = 100
        frequency_A = 100
        frequency_G = 100
        frequency_C = 100
        frequency_T_0 = 100
        for _ in range(count_of_epochs):
            A_to_G -= step * self.calc_derivative_for_one_param_GTR(A_to_G, A_to_C, A_to_T, G_to_C, G_to_T, C_to_T, frequency_A, frequency_G, frequency_C, frequency_T_0, check_param = 'A_to_G')
            A_to_C -= step * self.calc_derivative_for_one_param_GTR(A_to_G, A_to_C, A_to_T, G_to_C, G_to_T, C_to_T, frequency_A, frequency_G, frequency_C, frequency_T_0, check_param = 'A_to_C')
            A_to_T -= step * self.calc_derivative_for_one_param_GTR(A_to_G, A_to_C, A_to_T, G_to_C, G_to_T, C_to_T, frequency_A, frequency_G, frequency_C, frequency_T_0, check_param = 'A_to_T')
            G_to_C -= step * self.calc_derivative_for_one_param_GTR(A_to_G, A_to_C, A_to_T, G_to_C, G_to_T, C_to_T, frequency_A, frequency_G, frequency_C, frequency_T_0, check_param = 'G_to_C')
            G_to_T -= step * self.calc_derivative_for_one_param_GTR(A_to_G, A_to_C, A_to_T, G_to_C, G_to_T, C_to_T, frequency_A, frequency_G, frequency_C, frequency_T_0, check_param = 'G_to_T')
            C_to_T -= step * self.calc_derivative_for_one_param_GTR(A_to_G, A_to_C, A_to_T, G_to_C, G_to_T, C_to_T, frequency_A, frequency_G, frequency_C, frequency_T_0, check_param = 'C_to_T')
            frequency_A -= step * self.calc_derivative_for_one_param_GTR(A_to_G, A_to_C, A_to_T, G_to_C, G_to_T, C_to_T, frequency_A, frequency_G, frequency_C, frequency_T_0, check_param = 'frequency_A')
            frequency_G -= step * self.calc_derivative_for_one_param_GTR(A_to_G, A_to_C, A_to_T, G_to_C, G_to_T, C_to_T, frequency_A, frequency_G, frequency_C, frequency_T_0, check_param = 'frequency_G')
            frequency_C -= step * self.calc_derivative_for_one_param_GTR(A_to_G, A_to_C, A_to_T, G_to_C, G_to_T, C_to_T, frequency_A, frequency_G, frequency_C, frequency_T_0, check_param = 'frequency_C')
            frequency_T_0 -= step * self.calc_derivative_for_one_param_GTR(A_to_G, A_to_C, A_to_T, G_to_C, G_to_T, C_to_T, frequency_A, frequency_G, frequency_C, frequency_T_0, check_param = 'frequency_T')
            print(self.calc_derivative_for_one_param_GTR(A_to_G, A_to_C, A_to_T, G_to_C, G_to_T, C_to_T, frequency_A, frequency_G, frequency_C, frequency_T_0, check_param = 'frequency_A'))
        result_delta = self.calc_delta_matrix_GTR(A_to_G, A_to_C, A_to_T, G_to_C, G_to_T, C_to_T, frequency_A, frequency_G, frequency_C, frequency_T_0)
        return [[A_to_G, A_to_C, A_to_T, G_to_C, G_to_T, C_to_T, frequency_A, frequency_G, frequency_C, frequency_T_0], result_delta]




    def calc_llf_JC69(self, params_jc69):
        q_matrix = self.models_of_evolution_class.JC69(*params_jc69[0])
        p_matrix = scipy.linalg.expm(q_matrix * self.time)
        counter = 0
        i = 0
        while i < len(p_matrix):
            x = 0
            while x < len(p_matrix[i]):
                res = scipy.stats.poisson.pmf(self.emperical_probability_matrix[i][x],mu=self.to_divisions[i]*p_matrix[i][x])
                if res == 0.0:
                    res = 0.00000001
                counter += math.log(res)
                #print(res, math.log(res), counter, self.emperical_probability_matrix[i][x], self.to_divisions[i])
                #print(q_matrix)
                #print(p_matrix)
                x += 1
            i += 1
        return counter
    def calc_llf_K81(self, params_k81):
        q_matrix = self.models_of_evolution_class.K81(*params_k81[0])
        p_matrix = scipy.linalg.expm(q_matrix * self.time)
        counter = 0
        i = 0
        while i < len(p_matrix):
            x = 0
            while x < len(p_matrix[i]):
                res = scipy.stats.poisson.pmf(self.emperical_probability_matrix[i][x],mu=self.to_divisions[i]*p_matrix[i][x])
                if res == 0.0:
                    res = 0.00000001
                counter += math.log(res)
                x += 1
            i += 1
        return counter
    def calc_llf_F81(self, params_f81):
        q_matrix = self.models_of_evolution_class.F81(*params_f81[0])
        p_matrix = scipy.linalg.expm(q_matrix * self.time)
        counter = 0
        i = 0
        while i < len(p_matrix):
            x = 0
            while x < len(p_matrix[i]):
                res = scipy.stats.poisson.pmf(self.emperical_probability_matrix[i][x],mu=self.to_divisions[i]*p_matrix[i][x])
                if res == 0.0:
                    res = 0.00000001
                counter += math.log(res)
                x += 1
            i += 1
        return counter
    def calc_llf_GTR(self, params_gtr):

        q_matrix = self.models_of_evolution_class.GTR(*params_gtr[0])
        p_matrix = scipy.linalg.expm(q_matrix * self.time)
        counter = 0
        i = 0
        while i < len(p_matrix):
            x = 0
            while x < len(p_matrix[i]):
                res = scipy.stats.poisson.pmf(self.emperical_probability_matrix[i][x],mu=self.to_divisions[i]*p_matrix[i][x])
                if res == 0.0:
                    res = 0.00000001
                counter += math.log(res)
                x += 1
            i += 1
        return counter

    def predict(self, print_datas=True):
        self.emperical_intensity_matrix, self.emperical_probability_matrix = self.calc_emperical_matrix()
        params_jc69 = self.calc_best_params_for_JC69()
        params_k81 = self.calc_best_params_for_K81()
        params_f81 = self.calc_best_params_for_F81()
        params_gtr = self.calc_best_params_for_GTR()
        self.emperical_probability_matrix = self.emperical_probability_matrix * self.to_divisions
        llf_jc69 = self.calc_llf_JC69(params_jc69)
        llf_k81 = self.calc_llf_K81(params_k81)
        llf_f81 = self.calc_llf_F81(params_f81)
        llf_gtr = self.calc_llf_GTR(params_gtr)
        params_jc69[1] = 2 * (1 - llf_jc69)
        params_k81[1] = 2 * (3 - llf_k81)
        params_f81[1] = 2 * (4 - llf_f81)
        params_gtr[1] = 2 * (10 - llf_gtr)

        if print_datas:
            print('Empirical P-matrix: ')
            print(self.emperical_probability_matrix / self.to_divisions)
            print('Empirical Q-matrix: ')
            print(self.emperical_intensity_matrix)
            print(f'Best params for JC69: {params_jc69[0]}')
            print(f'Best params for K81: {params_k81[0]}')
            print(f'Best params for F81: {params_f81[0]}')
            print(f'Best params for GTR: {params_gtr[0]}')
            print(f'LLF(loglikelihood) for JC69: {llf_jc69}')
            print(f'LLF(loglikelihood) for K81: {llf_k81}')
            print(f'LLF(loglikelihood) for F81: {llf_f81}')
            print(f'LLF(loglikelihood) for GTR: {llf_gtr}')
            print(f'Difference points for JC69 method taking into account the complexity of the model: {params_jc69[1]}')
            print(f'Difference points for K81 method taking into account the complexity of the model: {params_k81[1]}')
            print(f'Difference points for F81 method taking into account the complexity of the model: {params_f81[1]}')
            print(f'Difference points for GTR method taking into account the complexity of the model: {params_gtr[1]}')
        min_model = min(params_jc69[1], params_k81[1], params_f81[1], params_gtr[1])
        if params_jc69[1] == min_model:
            if print_datas:
                print('Predicted model: JC69')
                print(f'Best params: {params_jc69[0]}')
            return 0
        if params_k81[1]  == min_model:
            if print_datas:
                print('Predicted model: K81')
                print(f'Best params: {params_k81[0]}')
            return 1
        if params_f81[1]  == min_model:
            if print_datas:
                print('Predicted model: F81')
                print(f'Best params: {params_f81[0]}')
            return 2
        if params_gtr[1]  == min_model:
            if print_datas:
                print('Predicted model: GTR')
                print(f'Best params: {params_gtr[0]}')
            return 3
        return 4 #bug: for some reason, the P_matrix calculated in llf_functions may have negative values for small times, in which case pmf return nan
if __name__ == "__main__":
    path_to_old_sequence = input('Enter path to fasta file with first (old) sequence: ')
    path_to_young_sequence = input('Enter path to fasta file with second (young) sequence: ')
    time = float(input('Enter time: '))
    for record in SeqIO.parse(path_to_old_sequence, "fasta"):
        old_sequence = record.seq 
    for record in SeqIO.parse(path_to_young_sequence, "fasta"):
        new_sequnce = record.seq        
    mep = Model_Evolution_Predictor(old_sequence, new_sequnce, time)
    try:
        num_of_model_predicted = mep.predict()
    except:
        print('Error detected. Maybe you should change the time?')
