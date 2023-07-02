import numpy as np
import random
class Models_Evolution():
    def JC69(self, alpha):
        intensity_matrix = np.array(  [[-3*alpha, alpha, alpha, alpha],
                                       [alpha, -3*alpha, alpha, alpha],
                                       [alpha, alpha, -3*alpha, alpha],
                                       [alpha, alpha, alpha, -3*alpha]])
        return intensity_matrix

    def K81(self, alpha, beta, gamma):
        diag = -1 * (alpha + beta + gamma)
        intensity_matrix = np.array([[diag,  alpha, beta,  gamma],
                                       [alpha, diag,  gamma, beta],
                                       [beta,  gamma, diag,  alpha],
                                       [gamma, beta,  alpha, diag]])
        return intensity_matrix
    def F81(self, frequency_A, frequency_G, frequency_C, frequency_T):
        diag_A = -1 * (frequency_G + frequency_C + frequency_T)
        diag_G = -1 * (frequency_A + frequency_C + frequency_T)
        diag_C = -1 * (frequency_A + frequency_G + frequency_T)
        diag_T = -1 * (frequency_A + frequency_G + frequency_C)
        intensity_matrix = np.array([[diag_A,      frequency_G, frequency_C, frequency_T],
                                       [frequency_A, diag_G,      frequency_C, frequency_T],
                                       [frequency_A, frequency_G, diag_C,      frequency_T],
                                       [frequency_A, frequency_G, frequency_C, diag_T]])
        return intensity_matrix
    def SYM(self, A_to_G, A_to_C, A_to_T, G_to_C, G_to_T, C_to_T):
        A_to_A = -1 * (A_to_T + A_to_C + A_to_G)
        T_to_T = -1 * (A_to_T + C_to_T + G_to_T)
        C_to_C = -1 * (A_to_C + C_to_T + G_to_C)
        G_to_G = -1 * (A_to_G + G_to_T + G_to_C)
        intensity_matrix = np.array([[A_to_A, A_to_G, A_to_C, A_to_T],
                                       [A_to_G, G_to_G, G_to_C, G_to_T],
                                       [A_to_C, G_to_C, C_to_C, C_to_T],
                                       [A_to_T, G_to_T, C_to_T, T_to_T]])
        return intensity_matrix
    def generate_random_intensity_matrix(self, rand_value='True', print_datas=True):
        if rand_value == 'True':
            rand_value = random.randint(0,3)
        if rand_value == 0:
            params = [1/random.randint(1,100)]
            #params = [0.5]
            if print_datas:
                print(f'Start model: JC69, with params = {params}')
            intensity_matrix = self.JC69(*params)
        elif rand_value == 1:
            #params = [0.2,0.3,0.5]
            params = [1/random.randint(1,100),1/random.randint(1,100),1/random.randint(1,100)]
            if print_datas:
                print(f'Start model: K81, with params = {params}')
            intensity_matrix =  self.K81(*params)
        elif rand_value == 2:
            #params = [0.2,0.4,0.5,0.8]
            params = [1/random.randint(1,100),1/random.randint(1,100),1/random.randint(1,100),1/random.randint(1,100)]
            if print_datas:
                print(f'Start model: F81, with params = {params}')
            intensity_matrix = self.F81(*params)
        else:
            params = [1/random.randint(1,100),1/random.randint(1,100),1/random.randint(1,100),1/random.randint(1,100),1/random.randint(1,100),1/random.randint(1,100)]
            if print_datas:
                print(f'Start model: SYM, with params = {params}')
            intensity_matrix = self.SYM(*params)
        return rand_value, intensity_matrix
