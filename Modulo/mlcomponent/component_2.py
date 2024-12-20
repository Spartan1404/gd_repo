import pickle as pck
import random as ram
from MyBaseHechos import calculateAll
from MyBaseHechos import managed_load
from enfoqueMedia.pruebaEnf25 import *
from enfoqueMedia.pruebaEnf50 import *
from enfoqueMedia.pruebaEnf100 import *
from enfoqueMedia.pruebaEnf_200 import *
from enfoqueMedia.pruebaEnf_300 import *

class Component:

    def __init__(self, e='3'):
        self.x_instances = []
        self.y_instances = []
        self.x_positives = []
        self.e = e
        self.positive_reclasifcation = []
        self.metrics_characterization = []
        self.dt = None
        self.characterization_database = None

    def validate(self, tipe, x_new=[], y_new=[]):
        if tipe == 1:
            if len(y_new) == 0 or len(x_new) == 0:
                print("New data array is empty")
                return False
        if tipe == 2:
            if len(self.x_instances) == 0:
                print("Array data is empty")
                return False
            if len(self.x_positives) == 0:
                print("No human user data exists")
                return False
        if tipe == 3:
            if len(self.metrics_characterization) == 0:
                print("Lista de de metricas de caracterización vacia")
                return False

        return True

    def show(self, lista):
        uniq, count = np.unique(lista, return_counts=True)
        try:
            print("Number of human users in reclassification:", count[0])
        except:
            print("Number of human users in reclassification:", 0)
        try:
            print("Number of bots users in reclassification:", count[1])
        except:
            print("Number of bots users in reclassification:", 0)

    def split_x_y(self, instances):
        self.x_instances = []
        self.y_instances = []
        for i in instances:
            self.x_instances.append(i[:-1])
            self.y_instances.append(i[-1])
        self.x_instances = np.array(self.x_instances)
        self.y_instances = np.array(self.y_instances)

    def add_data(self, x_new, y_new):  # anade las instancias clasificadas en pf al fichero
        data_new = []
        for i, j in zip(x_new, y_new):
            data_new.append(np.r_[i, [j]])

        data_new = np.array(data_new)
        self.save_data_new(data_new)

    def load_file_instances(self):
        file = open("./files/file_clasf_pf_bueno.pckl", "ab+")
        file.seek(0)
        inst = np.array([])
        try:
            inst = pck.load(file)
            self.split_x_y(inst)
        except:
            print("Created file of instances classified by proactive forest")
        finally:
            file.close()
        return inst

    def save_data_new(self, data_new):
        instances_file = self.load_file_instances()
        try:
            data_new = np.r_[instances_file, data_new]
        except:
            pass
        file = open("./files/file_clasf_pf_bueno.pckl", "wb")
        pck.dump(data_new, file)
        file.close()

    def simulate_positives_2(self, x, porcent_cant_min):
        """
        Altera los datos clasificados como humanos dentro de un rango especificado sin introducir aleatoriedad.

        Args:
            x: Matriz de datos a alterar.
            porcent_cant_min: Porcentaje mínimo de datos a alterar.
            porcent_cant_max: Porcentaje máximo de datos a alterar.

        Returns:
            Matriz de datos alterada.
        """

        # Calcular el número de datos a alterar
        cant = len(x)
        cant_atributes = x.shape[1]
        cant_modif = int(porcent_cant_min * cant)
        print("Cantidad de instancias modificadas", cant_modif)

        # Obtener el rango de valores para la alteración
        max_value = np.max(self.x_instances)
        min_value = np.min(self.x_instances)
        range_value = max_value - min_value

        # Alterar los datos
        for index in range(cant_modif):
            # Seleccionar una columna y calcular el nuevo valor basado en el índice
            colum = index % cant_atributes
            new_value = min_value + (index / cant_modif) * range_value

            # Alterar el valor en la matriz de datos
            x[index][colum] = new_value

        return x

    def set_positives(self, x_clasf, y_clasf):  # metodo que devuelve las instancias clasificadas como humanos
        print('Eliminando instancias de bots de los datos de entrada...')
        print(f'Total de datos: {len(x_clasf)}')
        for i, j in zip(x_clasf, y_clasf):
            if j == 0:
                self.x_positives.append(i)
        print('Instancias de bots eliminadas')
        print(f'Total de datos humanos: {len(self.x_positives)}')
        print('Ejecutando simulacion de cambio de comportamiento...')
        self.x_positives = self.simulate_positives_2(np.array(self.x_positives), 0.4)


    def set_characterization_label(self):  # metodo para asignar la etiqueta a la caracterizacion del conjunto de instancias
        if self.validate(3):
            numeros = [25000, 50000, 100000, 200000, 300000]
            num = len(self.x_positives)
            valor_cercano = None
            min_diferencia = float('inf')
            for num_lista in numeros:
                diferencia = abs(num - num_lista)
                if diferencia < min_diferencia:
                    min_diferencia = diferencia
                    valor_cercano = num_lista
            result = None
            match valor_cercano:
                case 25000:
                    result = pruebaEnfoque_i25_tp20(self.x_positives)
                case 50000:
                    result = pruebaEnfoque_i50_tp20(self.x_positives)
                case 100000:
                    result = pruebaEnfoque_i100_tp25(self.x_positives)
                case 200000:
                    result = pruebaEnfoque_i200_tp5(self.x_positives)
                case 300000:
                    result = pruebaEnfoque_i300_tp25(self.x_positives)
                case _:
                    print('default')
            if result is True:
                print('Etiqueta asignada: bots')
                self.metrics_characterization.append(1)
            else:
                print('Etiqueta asignada: no bots')
                self.metrics_characterization.append(0)

    def refresh_characterization_database(self, row):  # metodo que actualiza una fila de la base de hechos
        self.characterization_database = open("./files/characterization_database.txt", "r+")
        lines = self.characterization_database.readlines()
        for index, l in enumerate(lines):
            if l[:-2] == row[:-1] and l[-2] != row[-1]:
                l = l[:-2]
                l += row[-1] + "\n"
                lines[index] = l
                self.characterization_database.seek(0)
                self.characterization_database.writelines(lines)
                print("Updated row to database")
                self.characterization_database.close()
                return True
            elif l[:-2] == row[:-1] and l[-2] == row[-1]:
                print("Data already exist")
                return True
        self.characterization_database.close()
        return False

    def add_row(self, row):  # metodo que anade una fila a la base de hechos
        self.characterization_database = open("./files/characterization_database.txt", "a")
        self.characterization_database.write(row + "\n")
        self.characterization_database.close()

    def save_data_characterization(
            self):  # metodo para salvar la caracterizacion del cojunto de datos en la base de hechos
        if self.validate(3):
            row = ""
            for i in self.metrics_characterization:
                row += str(i) + ";"
            row = row[:-1]
            print(row)
            try:
                if not self.refresh_characterization_database(row):
                    self.add_row(row)
                    print("Added row to database")
            except:
                print("Data base is created")
                self.add_row(row)
                print("Added row to database")

    def calculate_metrics(self):  # se claculan las metricas
        if self.validate(2):
            normaldata, normalLabel3 = managed_load(since=0, untilBot=0, untilHuman=len(self.x_positives), e='0')
            self.metrics_characterization = calculateAll(self.x_positives, normaldata, self.e)


    def run_charact(self, x_clasf, y_clasf):
        if len(x_clasf) == 0 or len(y_clasf) == 0:
            print("The users array is empty")
        else:
            print('Inicializando meta-componente\n')
            self.set_positives(x_clasf, y_clasf)  # De todos los datos que entraron nos quedamos con los humanos y después se simula un cambio
            print('Calculando metricas\n')
            self.calculate_metrics()  # se calculan las metricas para describir el conjunto de datos
            print('Corriendo el enfoque para asignar la etiqueta\n')
            self.set_characterization_label()  # Se le asigna la etiqueta al conjunto de datos usando el enfoque de entropia
            print('Anadiendo fila a la base de hecho\n')
            self.save_data_characterization()  # se guarda la nueva fila en la base de hechos
