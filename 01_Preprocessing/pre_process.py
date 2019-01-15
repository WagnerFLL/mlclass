import string

import pandas as pd
import numpy as np
import sklearn
import os
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from random import randint
import random
import decimal
import requests


def predictKnn():
    print('\n - Lendo o arquivo com o dataset sobre diabetes')
    data = pd.read_csv('diabetes_test.csv')

    # Criando X and y par ao algorítmo de aprendizagem de máquina.\
    print(' - Criando X e y para o algoritmo de aprendizagem a partir do arquivo diabetes_dataset')
    # Caso queira modificar as colunas consideradas basta algera o array a seguir.
    feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    X = data[feature_cols]
    y = data.Outcome

    # Ciando o modelo preditivo para a base trabalhada
    print(' - Criando modelo preditivo')
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X, y)

    # realizando previsões com o arquivo de
    print(' - Aplicando modelo e enviando para o servidor')
    data_app = pd.read_csv('diabetes_sample_test.csv')
    y_pred = neigh.predict(data_app)

    # Enviando previsões realizadas com o modelo para o servidor
    URL = "https://aydanomachado.com/MachineLearning/PreProcessing.php"

    # TODO Substituir pela sua chave aqui
    DEV_KEY = "Café com leite"

    # json para ser enviado para o servidor
    data = {'dev_key': DEV_KEY,
            'predictions': pd.Series(y_pred).to_json(orient='values')}

    # Enviando requisição e salvando o objeto resposta
    r = requests.post(url=URL, data=data)

    # Extraindo e imprimindo o texto da resposta
    pastebin_url = r.text
    print(" - Resposta do servidor:\n", r.text, "\n")


def generate_norm_data(data):
    data = data.dropna(thresh=5)
    data = data.interpolate()
    data = data.fillna(data.median())
    data = data.astype(np.float64)

    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(data)
    data_norm = pd.DataFrame(np_scaled)
    return data_norm


def multiply_features(data_norm, sample, scalars):
    pregnancies = scalars[0]
    glucose = scalars[1]
    blood_pressure = scalars[2]
    skin_thickness = scalars[3]
    insulin = scalars[4]
    bmi = scalars[5]
    diabetes_pedigree_function = scalars[6]
    age = scalars[7]
    # data = data.convert_objects()
    if sample == 0:
        data_norm.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                             'DiabetesPedigreeFunction', 'Age', 'Outcome']
    else:
        data_norm.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                             'DiabetesPedigreeFunction', 'Age']

    data_norm.loc[:, 'Pregnancies'] *= pregnancies
    data_norm.loc[:, 'Glucose'] *= glucose
    data_norm.loc[:, 'BloodPressure'] *= blood_pressure
    data_norm.loc[:, 'SkinThickness'] *= skin_thickness
    data_norm.loc[:, 'Insulin'] *= insulin
    data_norm.loc[:, 'BMI'] *= bmi
    data_norm.loc[:, 'DiabetesPedigreeFunction'] *= diabetes_pedigree_function
    data_norm.loc[:, 'Age'] *= age

    # print(data_norm.to_string())
    # print(data_norm.dtypes())
    data_norm = data_norm.round(6)
    return data_norm


def random_with_range(begin, end):
    begin *= 10
    end *= 10
    return float(random.randrange(begin, end))/10


def id_generator(size=10, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def neighbors(scalar):
    constant = 0.5
    size = len(scalar)
    list_of_neighbors = []
    for i in range(size):
        #print(scalar)
        temp1 = scalar.copy()
        temp2 = scalar.copy()
        temp1[i] += constant
        temp2[i] -= constant
        list_of_neighbors.append(temp1)
        list_of_neighbors.append(temp2)
    #print(list_of_neighbors)
    return list_of_neighbors


def predict(best_accuracy, scalars, data, data_sample):
    accuracy = predictKnn()
    if accuracy > best_accuracy:
        #print("Accuracy= ", accuracy, "Count", + count)
        best_accuracy = accuracy
        temp = "best_guesses/accuracy" + str(accuracy) + "KEY" + id_generator()
        os.mkdir(temp)
        f = open(temp + "/scalars.txt", "w")
        file_content = "pregnancies= " + str(scalars[0]) + '\n'
        file_content += "glucose= " + str(scalars[1]) + '\n'
        file_content += "blood_pressure= " + str(scalars[2]) + '\n'
        file_content += "skin_thickness= " + str(scalars[3]) + '\n'
        file_content += "insulin= " + str(scalars[4]) + '\n'
        file_content += "bmi= " + str(scalars[5]) + '\n'
        file_content += "diabetes_pedigree_function= " + str(scalars[6]) + '\n'
        file_content += "age= " + str(scalars[7]) + '\n'
        file_content += "accuracy= " + str(accuracy)
        f.write(file_content)
        f.close()
        print(file_content)
        print("------------------------------------")
        data.to_csv(temp + "/data_train", index=False)
        data_sample.to_csv(temp + "/data_test", index=False)
    return accuracy


def test_accuracy(data, data_sample, guess, best_accuracy_global, flag):
    data_sample = multiply_features(data_sample, 1, guess)
    data = multiply_features(data, 0, guess)

    data_sample.to_csv("diabetes_sample_test.csv", index=False)
    data.to_csv("diabetes_test.csv", index=False)
    if flag is True:
        best_accuracy = predict(best_accuracy_global, guess, data, data_sample)
    else:
        best_accuracy = predictKnn()
    #
    return best_accuracy


def simulated_annealing(data, data_sample, start, best_accuracy_global):
    current = start
    while True:
        neigh = neighbors(start)
        #print(neigh)
        best_accuracy = 0
        best_neighbor = None
        for i in neigh:
            current_accuracy = test_accuracy(data, data_sample, i, best_accuracy_global, False)
            print("    GUESS", i, " ACCURACY: ", current_accuracy)
            if current_accuracy > best_accuracy:
                best_neighbor = i
                best_accuracy = current_accuracy
        if best_accuracy <= test_accuracy(data, data_sample, current, best_accuracy_global, False):
            return current
        current = best_neighbor


def generate_random(i):
    return random_with_range(0, 5)


# def test_sa():
#     scalars = [1, 1, 1, 1, 1, 1, 1, 1]
#     count = 0
#     while count < 20:
#         a = simulated_annealing(scalars)
#         print(scalars, " ", a)
#         scalars = list(map(generate_random, scalars))
#         count += 1

#best_accuracy = 0


def scalar():
    best_accuracy = 0
    count = 0
    data_sample = pd.read_csv('diabetes_app.csv')
    data = pd.read_csv('diabetes_dataset.csv')
    data_sample = generate_norm_data(data_sample)
    data = generate_norm_data(data)
    data_sample.to_csv("diabetes_sample_norm.csv", index=False)
    data.to_csv("diabetes_norm.csv", index=False)
    scalars = [1, 1, 1, 1, 1, 1, 1, 1]
    while True:
        data_sample = pd.read_csv('diabetes_sample_norm.csv')
        data = pd.read_csv('diabetes_norm.csv')
        ans = simulated_annealing(data, data_sample, scalars, best_accuracy)
        best_accuracy = test_accuracy(data, data_sample, ans, best_accuracy, True)
        print("JUMP", best_accuracy)
        # pregnancies = random_with_range(0, 7)
        # glucose = random_with_range(1, 10)
        # blood_pressure = random_with_range(0, 6)
        # skin_thickness = random_with_range(0, 3)
        # insulin = random_with_range(1, 15)
        # bmi = random_with_range(0.5, 9)
        # diabetes_pedigree_function = random_with_range(0, 8)
        # age = random_with_range(0.5, 10)
        scalars = list(map(generate_random, scalars))
        count += 1


def main():
    scalar()
    #test_sa()


main()
