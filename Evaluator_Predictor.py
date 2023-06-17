from Models_Evolution import Models_Evolution
from Engine_Evolution import Engine_Evolution
from Predictor_Model_Evolution import Model_Evolution_Predictor
import random
import numpy as np
from tqdm import tqdm
class Evaluator_Predictor():
    def __init__(self):
        self.sequence_description = 'TMP'
        self.sequence = ''.join(random.choice('ATCG') for i in range(10**6))
        self.models_evolution = Models_Evolution()

    def check_predictor_by_params(self, time, count_of_attempts=100):
        '''
        Real\Pred  JC69   K81  F81  GTR  ERRORS
        JC69
        K81
        F81
        GTR
        '''
        matrix_of_errors = np.array([[0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0]])
        for i in range(0,4):
            for _ in tqdm(range(count_of_attempts)):
                num_of_model, intensity_matrix = self.models_evolution.generate_random_intensity_matrix(rand_value=i, print_datas=False)
                engine = Engine_Evolution(self.sequence, time, intensity_matrix, print_datas = False)
                new_sequnce = engine.get_new_sequence()
                mep = Model_Evolution_Predictor(self.sequence, new_sequnce, time)
                try:
                    num_of_model_predicted = mep.predict(print_datas=False)
                except:
                    num_of_model_predicted = 4
                matrix_of_errors[num_of_model][num_of_model_predicted] += 1
                #print(num_of_model, num_of_model_predicted, matrix_of_errors)

        return matrix_of_errors
    def evaluate(self, time_for_check=None):
        print('''General view of the error matrix:
        Real\Pred  JC69   K81  F81  GTR  ERRORS
        JC69        N      N    N    N     N
        K81         N      N    N    N     N
        F81         N      N    N    N     N
        GTR         N      N    N    N     N
        ''')

        if time_for_check == None:
            time_for_check = [3]
        for time in time_for_check:
            matrix_of_errors = self.check_predictor_by_params(time)
            print(f'For time = {time}, the error matrix looks like this: ')
            print(matrix_of_errors)
if __name__ == "__main__":
    #it won't be fast
    time_for_check = input('Enter times for check (ex: 0.5, 0.1, ...) or enter STANDART for standart check: ')
    if time_for_check == "STANDART":
        time_for_check = [1e-05, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
    else:
        t_arr = []
        for x in time_for_check.split(', '):
            t_arr.append(float(x))
        time_for_check = t_arr
    checker = Evaluator_Predictor()
    checker.evaluate(time_for_check=time_for_check )
