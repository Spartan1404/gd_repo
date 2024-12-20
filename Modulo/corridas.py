from Facade import *
from utils import utils, LoadData

def correr():
    for e in range(2, 13):
        if e == 3 or e == 11:
            continue

        print('Scenary: ' + str(e))

        for i in range(10):
            with open('resultados.txt', 'a') as f:
                print(f'\nEscenario {e}, iteracion {i}\n', file=f)
            model = fit_pf_process(esc=str(e))


if __name__ == '__main__':
    correr()
