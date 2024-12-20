import pandas as pd
import os

def start():    
    for e in range(13):
        num = e + 1
        if 1 <= num <= 13:
            df = pd.read_csv(f'./database/{num}.binetflow', sep=',')
        else:
            print("Opción inválida")

        bot = 1000000#int(input("Number of bots you want to use: "))
        notBot = 90000#int(input("Number of Not bots you want to use: "))

        # HTTP : sera usado para CF , US y HTTP para estos tipos de ataque revisar tabla pagina oficial de ctu-13
        # PortScan(PS) : TCP
        Escenarios = {1: ["IRC", "SPAM", "HTTP"],
                      2: ["IRC", "SPAM", "HTTP"],
                      3: ["IRC", "TCP", "HTTP"],
                      4: ["IRC", "DNS", "HTTP"],
                      5: ["SPAM", "TCP", "HTTP"],
                      6: ["TCP"],
                      7: ["HTTP"],
                      8: ["TCP"],
                      9: ["IRC", "SPAM", "HTTP", "TCP"],
                      10: ["IRC", "DNS", "HTTP"],
                      11: ["IRC", "DNS", "HTTP"],
                      12: ["P2P"],
                      13: ["SPAM", "TCP", "HTTP"]}

        df1_list = []  # Lista para almacenar los dataframes df1

        # Verifica si la clave es igual al número ingresado por el usuario
        if num in Escenarios:
            bot = bot = int(bot / len(Escenarios[num]))
            print(f"La clave {num} tiene {len(Escenarios[num])} elementos.")
            for value in Escenarios[num]:  # Itera sobre los valores de la clave
                condicion1 = df['Label'].str.contains('Botnet', case=False) & df['Label'].str.contains(value,
                                                                                                       case=False)
                print(bot)
                df1 = df[condicion1]  # Filtra el dataframe original con la condición1.
                df1 = df1.iloc[
                      :bot]  # Selecciona las primeras  filas del dataframe df1 segun el numero que tenga la variable 'bot'.
                df1_list.append(df1)  # Añade el dataframe df1 a la lista df1_list
        else:
            print("La clave no existe en el diccionario.")

        condicion2 = df['Label'].str.contains('Normal', case=False) | df['Label'].str.contains('Background', case=False)
        df2 = df[condicion2]
        df2 = df2.iloc[:notBot]

        df = pd.concat(df1_list + [df2])

        if not os.path.isfile('0.binetflow'):
            df.to_csv('0.binetflow', index=False)
        else:
            df.to_csv('0.binetflow', mode='a', header=False, index=False)

if __name__ == "__main__":
    start()






# antes del desastre
# import pandas as pd
# import os

# def start():    
#     print("\nPick your option to continue")
#     print("-------------------------------------")
#     print("1. Scenario 1 ")
#     print("2. Scenario 2 ")
#     print("3. Scenario 3 ")
#     print("4. Scenario 4 ")
#     print("5. Scenario 5 ")
#     print("6. Scenario 6 ")
#     print("7. Scenario 7 ")
#     print("8. Scenario 8 ")
#     print("9. Scenario 9 ")
#     print("10. Scenario 10 ")
#     print("11. Scenario 11 ")
#     print("12. Scenario 12 ")
#     print("13. Scenario 13 ")
#     print("------------------------------------\n")

#     print('Scenarios range from 1 to 13\n')
#     valid_option = False
    
#     while not valid_option:
#         option = input("Pick the scenery you want to use: ")
#         if option.isdigit():
#             num = int(option)
#             if 1 <= num <= 13:
#                 df = pd.read_csv(f'./database/{num}.binetflow', sep=',')
#                 valid_option = True
#             else:
#                 print("Opción inválida")
#         else:
#             print("Por favor, introduce un número")

#     bot = int(input("Number of bots you want to use: "))
#     notBot = int(input("Number of Not bots you want to use: "))    

#     condicion1 = df['Label'].str.contains('Botnet', case=False)
#     condicion2 = df['Label'].str.contains('Normal', case=False) | df['Label'].str.contains('Background',case=False)

#     df1 = df[condicion1]
#     df2 = df[condicion2]

#     df1 = df1.iloc[:bot]
#     df2 = df2.iloc[:notBot]

#     df = pd.concat([df1, df2])

#     if not os.path.isfile('BaseCentral.csv'):
#         df.to_csv('BaseCentral.csv', index=False)
#     else: 
#         df.to_csv('BaseCentral.csv', mode='a', header=False, index=False)

# if __name__ == "__main__":
#     start()


#     # Lee el archivo .binetflow
#     df = pd.read_csv('1.binetflow', sep=',')

#     # Crea una condición para cada fila
#     condicion1 = df['Label'].str.contains('Botnet', case=False)
#     condicion2 = df['Label'].str.contains('Normal', case=False)| df['Label'].str.contains('BAckground',case=False)

#     # Selecciona solo las filas que cumplen la primera condición
#     df1 = df[condicion1]

#     # Selecciona solo las filas que cumplen la segunda condición
#     df2 = df[condicion2]

#     # Selecciona solo las n primeras líneas de cada DataFrame
#     df1 = df1.iloc[:bot]
#     df2 = df2.iloc[:notBot]

#     # Concatena los dos dataframes
#     df = pd.concat([df1, df2])

#     # Escribe el nuevo DataFrame en un archivo CSV
#     df.to_csv('nuevo_archivo.csv', index=False, mode="a")

# if __name__ == "__main__":
#     start(2,3)




