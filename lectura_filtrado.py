# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 11:54:33 2023

@author: hecto
"""

#Script para combinar todos los archivos .cvs y despues filtar

#librerias
import pandas as pd
import os
import numpy as np

#generamos una variable que tenga la ruta donde estan todas las carpetas 
#con los archivos .csv
ruta = 'C:/Users/hecto/OneDrive/Documentos/ITA/Posdoc Proyecto/articulo 1/codigos/base_datos/'

#generamos una variable que continene los nombres de las carpetas
carpetas = {0:'DV', 1:'HR', 2:'PM25', 3:'PP', 4:'TMP', 5:'VV'}

#combinamos los conjuntos de archivos .cvs, cada carpeta tiene 12 archivos
#lo que determina la información de 1 año

#damos de alta la ruta donde se van a guardar las compilaciones de cada uno de 
#de los parametros
ruta_compilado = 'C:/Users/hecto/OneDrive/Documentos/ITA/Posdoc Proyecto/articulo 1/codigos/base_datos/compilado/'

#damos de alta la ruta donde se guardara el archivo final con la combinación de
#todos los parametros
ruta_datos_fin = 'C:/Users/hecto/OneDrive/Documentos/ITA/Posdoc Proyecto/articulo 1/codigos/base_datos/datos_unidos/'

#este ciclo funciona para movernos entre las multiples carpetas de los parámetros
for j in range(len(carpetas)):
    
    #leemos todos los archivos de la carpeta
    archivos = os.listdir(ruta + carpetas[j])
    
    #en este ciclo nos vemos por cada uno de los archivos dentro de la carpeta
    for i in range(len(archivos)):
        #esta sección se evalua para el 1er df
        if i == 0:
            #leemos los archivos para combinarlos
            df = pd.read_csv(ruta + carpetas[j] + '/' + archivos[i])
            
            #cambiamos el nombre de la columna valor por el nombre del parámetro
            df.rename(columns={'Valor' : carpetas[j]}, inplace = True)
            
            #eliminamos las columnas que no nos interesan
            df.drop(columns = ['Parámetro', 'Unidad'], inplace = True)
        
        #esta sección se ejecuta para ir agregando la demás info al df inicial
        else:
            #leemos los otros df
            df_1 = pd.read_csv(ruta + carpetas[j] + '/' + archivos[i])
            
            #cambiamos el nombre de la columna valor por el nombre del parámetro
            df_1.rename(columns={'Valor' : carpetas[j]}, inplace = True)
            
            #eliminamos las columnas que no nos interesan
            df_1.drop(columns = ['Parámetro', 'Unidad'], inplace = True)
            
            #agregamos al df el df_1 utilizando la función concatenar para hacer
            #la lista más grande
            df = pd.concat([df, df_1], axis = 0)
        #esta sección se activa para guardar la info de todos los archivos
        #en la carpeta en un archivo que nombramos datos
        if i == len(archivos)-1:
            df.to_csv(ruta_compilado + carpetas[j] +'_datos.csv', index = False)
            
#ahora utilizamos todos los archivos generados para unirlos en uno solo pero
#conectados por las columnas y no por las filas como en la sección anterior

#leemos todos los archivos que estan en la carpeta
archivos = os.listdir(ruta_compilado)

#filtramos todos los que no tengan .csv
#archivos = [elemento for elemento in archivos if ".csv" in elemento]

#si en la carpeta ya esta generado el archivo de datos_final lo eliminamos de la lista
#para no generar un error con el siguiente ciclo
#archivos = [elemento for elemento in archivos if not 'datos_final.csv' in elemento]
#archivos = [elemento for elemento in archivos if not 'escalamiento.csv' in elemento]

#en este ciclo se pasa por todos los archivos generados anteriormente para unirlos todos
for i in range(len(archivos)):
    #leemos el 1er archivo
    if i == 0:
        #combinamos todos los archivos ahora si
        df = pd.read_csv(ruta_compilado + archivos[i])
        print(str(len(df)) + ' ' + archivos[i][:-4])
    
    #leemos los otros archivos para combinarlos
    else:
        #leemos el otro df
        df_1 = pd.read_csv(ruta_compilado + archivos[i])
        print(str(len(df_1)) + ' ' + archivos[i][:-4])
        
        #unimos los df con merge
        df = pd.merge(df, df_1, how = 'inner', on = ['Fecha', 'Hora'])
        #algunas veces se duplican las columnas asi que eliminamos
        df = df.drop_duplicates(subset=['Fecha','Hora'], keep="first")
    
    
#cambiamos la columna de la Hora de str a int, 1ro separamos 
#separamos tomando como referecnia ' - '
hora = df.Hora.str.split(':')

#seleccionamos solo el 1er elemento de la lista
hora = [i[0] for i in hora]

#cambiamos la columna de hora por la hora
df.Hora = hora

#la comvertimos de str a int
df.Hora = pd.to_numeric(df.Hora)

#genermos una nueva columna y convertimos la columna de Fecha a dia del año (YearD)
#y la extra a día de la semana (WeekD), pero 1ro la pasamos a forma to_datetime
df.Fecha = pd.to_datetime(df.Fecha)

#generamos la nueva columna
df = df.assign(YearD = np.nan)
df.YearD = df.Fecha

#cambiamos el nombre de la de fecha a día de la semana
df.rename(columns ={'Fecha' : 'WeekD'} ,inplace = True)

#haces los cambios
df.YearD = df.YearD.dt.dayofyear
df.WeekD = df.WeekD.dt.weekday

#generamos una nueva columna que tendra la concentración una hora anterior a la actual
# df = df.assign(YearD1 = np.nan)
# df = df.assign(Hora1 = np.nan)
# df = df.assign(PM25_t = np.nan)
# df.YearD1.iloc[1:len(df)] = df.YearD.iloc[0:len(df)-1]
# df.Hora1.iloc[1:len(df)] = df.Hora.iloc[0:len(df)-1]
# df.PM25_t.iloc[1:len(df)] = df.PM25.iloc[0:len(df)-1]

#eliminamos la 1ra fila del df
# df = df.drop(df.index[0])

# #eliminamos las filas que no complan con los siguientes criterios
# ciertos = []

# #con este ciclo buscamos los que se cumpla el criterio de los tiempo para la concentración a tiempo -1 hora
# for i in range(len(df)):
#     if df.Hora.iloc[i] - df.Hora1.iloc[i] == 1 or df.Hora.iloc[i] - df.Hora1.iloc[i] == -23:
#         if df.YearD1.iloc[i] == df.YearD.iloc[i] or df.YearD1.iloc[i] + 1 == df.YearD.iloc[i]:
#             ciertos.append(True)
#         else:
#             ciertos.append(False)
#     else:
#         ciertos.append(False)

# #eliminamos las columnas extras que servian como referencia para el filtrado
# df.drop(columns = ['YearD1', 'Hora1'], inplace = True)

# #aquí filtramos todos los elementos que no cumplieron con el critero
# df = df[ciertos]

#agregamos unas nuevas columnas con los datos de la c(t-1) y c(t-2)
df = df.assign(PM25_t = np.nan)
#df = df.assign(PM25_t2 = np.nan)

for i in range(len(df)-1):
    if df.Hora.iloc[i+1] - df.Hora.iloc[i] == 1 or df.Hora.iloc[i+1] - df.Hora.iloc[i] == -23:
        if df.YearD.iloc[i+1] == df.YearD.iloc[i] or df.YearD.iloc[i] + 1 == df.YearD.iloc[i+1]:
            df.PM25_t.iloc[i+1] = df.PM25.iloc[i]

df = df.drop(df.index[0])

# for i in range(len(df)-1):
#     if df.Hora.iloc[i+1] - df.Hora.iloc[i] == 1 or df.Hora.iloc[i+1] - df.Hora.iloc[i] == -23:
#         if df.YearD.iloc[i+1] == df.YearD.iloc[i] or df.YearD.iloc[i] + 1 == df.YearD.iloc[i+1]:
#             df.PM25_t2.iloc[i+1] = df.PM25_t.iloc[i] 

# df = df.drop(df.index[1])

#ahora filtramos los datos de concentraciones negativas
df = df[df.PM25 > 0]
df = df[df.PM25_t > 0]
#df = df[df.PM25_t2 > 0]

#guardamos el archivo final de esta estación de monitoreo
df.to_csv(ruta_datos_fin + 'datos_final.csv', index = False)