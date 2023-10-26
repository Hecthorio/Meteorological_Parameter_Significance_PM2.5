# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 13:25:35 2023

@author: hecto
"""

#script para generar una estimación del nivel de importancia de los parametros de
#entrada a una rna a partir del algortimo de Garson

#OJO! esto solo es una estomación, ya que el algotimo de garson utiliza los pesos
#la primera capa para el analisis, en redes profundas los pesos de las neuronas
#ya no se pueden definir por la influencia de los parametros de entrada de manera
#aislada, si no que desde la 1ra capa hacia adelante la información va mezclada por
#las evaluaciones de las funciones de activación de la red

#librerias que vamos a utilizar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
import seaborn as sns

###############################################################################
#                  LECTURA DE DATOS Y CARGAMOS EL MODELO                      #
###############################################################################

#damos de alta la ruta donde está el modelo de red neuronal
ruta_modelo = "C:/Users/hecto/OneDrive/Documentos/ITA/Posdoc Proyecto/articulo 1/codigos/modelos/"

#elegir el modelo
n = 7

#nombre del modelo
modelos ={0:'modelo_1.h5',
          1:'modelo_2.h5',
          2:'modelo_3.h5',
          3:'modelo_4.h5',
          4:'modelo_5.h5',
          5:'modelo_6.h5',
          6:'modelo_7.h5',
          7:'modelo_8.h5'}

#parametros de entrada a cada modelo
parametros = [['Time', 'WeekD', 'YearD'],
            ['Time', 'WeekD', 'YearD', 'TMP'],
            ['Time', 'WeekD', 'YearD', 'WS'],
            ['Time', 'WeekD', 'YearD', 'WD'],
            ['Time', 'WeekD', 'YearD', 'RH'],
            ['Time', 'WeekD', 'YearD', 'PP'],
            ['Time', 'WeekD', 'YearD', 'PM2.5(t-1)'],
            ['Time', 'WeekD', 'YearD', 'TMP', 'WS', 'WD', 'RH', 'PP', 'PM2.5(t-1)']]


#cargamos el modelo
red = load_model(ruta_modelo + modelos[n])

###############################################################################
#     EVALUACIÓN DE DERIVADAS PARCIALES PARA DETERMINAR LA SENSIBILIDAD       #
###############################################################################

#leemos la base de datos que se utilizo para el entrenamiento y validación de
#los modelos de redes neuronales
ruta_datos = "C:/Users/hecto/OneDrive/Documentos/ITA/Posdoc Proyecto/articulo 1/codigos/base_datos/datos_unidos/"
ruta_figuras = 'C:/Users/hecto/OneDrive/Documentos/ITA/Posdoc Proyecto/articulo 1/'

#damos de alta la función para el escalamiento de los datos
def escal_maxmin(X,min_max):
    X_esc = (X-min_max[1,:])/(min_max[0,:]-min_max[1,:])
    return X_esc

#lectura del compilado de base da todos filtrados
df = pd.read_csv(ruta_datos + 'datos_final.csv', header = 0)

#cambiamos de nombre los encabezados del df
df.rename(columns = {'Hora':'Time', 'DV':'WD', 'HR':'RH', 'PM25':'PM2.5(t)', 
                     'VV':'WS', 'PM25_t':'PM2.5(t-1)'}, inplace = True)

#leer los datos para el escalamiento
min_max = pd.read_csv(ruta_datos + 'escalamiento.csv', header = 0)

#eliminamos todas las filas donde la concentración sea 0
df = df[df['PM2.5(t)'] > 0]

#Separamos la columna de concentraciones y la eliminamos del df original
y_real = np.array(df['PM2.5(t)'])
df = df.drop(labels = ['PM2.5(t)'], axis=1)

#separamos solo los parametros que se usan en el modelo y los valores
#para hacer su escalamiento
x = np.array(df[parametros[n]])
escalar = np.array(min_max[parametros[n]])

#escalamos los valores de x
x = escal_maxmin(x, escalar)

#definimos nuestro parametro de incremento
delta = 0.0001

#evaluamos el modelo, f(x)
y = red.predict(x)

#generamos una variable donde guardamos las desviaciones estandar de cada conjunto de derivadas
std_todos = []
media_todos = []
RMSD_i = []

for m in range(len(parametros[n])):
    #generamos una copia de la matriz original de datos
    x_incre = x*1
    
    #agregamos un incremento en la variable iteradamente a cada columan para evaluar
    #las derivadas
    #OJO, el incremento para el día de la semana y el día del año se evalua diferente
    #porque son variables discretas
    if parametros[n][m] == 'WeekD' or parametros[n][m] == 'YearD':
        #agregamos el incremento de la variable discreta
        x_incre[:,m] = x_incre[:,m] + 1/max(min_max[parametros[n][m]])
        #guardamos el incremento para usarlo en la derivada después
        delta_dif = 1/max(min_max[parametros[n][m]])
        #si se pasa del intervalo de 1, regresa al valor cero solo si
        if parametros[n][m] == 'WeekD':
            x_incre[:,m][x_incre[:,m] > 1] = 0
    else:
        #agregamos el incremento de la variable continua
        x_incre[:,m] = x_incre[:,m] + delta
    
    #evaluar f(x+dx)
    y_incr = red.predict(x_incre)
    
    #evaluamos la derivada
    #OJO! La derivada de el día de la semana y el día del año se evaluan diferentes
    #porque son variables discretas
    if parametros[n][m] == 'WeekD' or parametros[n][m] == 'YearD':
        dcdx = (y_incr - y)/delta_dif
    else:
        dcdx = (y_incr - y)/delta
    
    #estandarizamos la derivada (las entradas ya estan estandarizadas) así
    #que solo vamos a estandarizar la salida (concentración)
    
    RMSD = (sum(dcdx**2)/len(dcdx))**0.5
    
    RMSD_i.append(RMSD)
    
    #evaluamos la media de la derivada (absoluta) y la desviación estandar
    media = np.zeros(len(dcdx))
    #media[:] = np.mean(abs(dcdx))*1
    media[:] = np.mean(dcdx)*1
    des_estan = np.std(dcdx)*1
    
    #guardamos cada valor de las desviaciones estandar
    std_todos.append(des_estan)
    media_todos.append(media[0])
    
    #graficamos para ver como son los cambios
    plt.figure()
    plt.plot(y_real, dcdx, 'o', color = 'black', markerfacecolor = 'none', label = 'Derivada')
    #plt.plot(y_real, media, '--', color = 'grey', label = 'Media')
    #plt.legend()
    plt.xlabel('Observed PM2.5 Concentration ($\mu g/m^3$)', fontsize = 16)
    plt.ylabel('$\partial y_{ann} / \partial x_i$', fontsize = 16)
    plt.xlim(0,max(y_real))
    plt.axhline(0, color = 'r', linestyle='--')
    if parametros[n][m] == 'WeekD':
        plt.title(parametros[n][m], fontsize = 18)
    else:
        plt.title(parametros[n][m], fontsize = 18)
    #esta sección es para acomodar la caja de texto de una manera estandarizada
    if max(dcdx) < 0:
        plt.text(max(y_real)*0.6, 0 - (0 - min(dcdx))*0.2, r'$std=$' + str(round(des_estan,3)) +
                 '\n$ mean = $' + str(round(media[0],3)), 
                 bbox={'facecolor': 'white', 'alpha': 0.8}, fontsize = 14)
    else:
        plt.text(max(y_real)*0.6, max(dcdx) - (max(dcdx)-min(dcdx))*0.2, r'$std=$' + str(round(des_estan,3)) +
             '\n$ mean = $' + str(round(media[0],3)), 
             bbox={'facecolor': 'white', 'alpha': 0.8}, fontsize = 14)
    plt.savefig(ruta_figuras + 'PaD_' + str(parametros[n][m]) + '.pdf', dpi = 300)

#generamos un grafico de barras que define la "importancia" de cada parametro
#en función de que tanta dispersión generó un cambio en ese parametro en la salida
#del modelo
pesos_std = std_todos/sum(std_todos)*100
pesos_media = media_todos/sum(media_todos)*100

#le cambiamos el nombre de la etiqueta Fecha a día
#parametros[n][0] = 'Día'

#graficamos el % importancia
plt.figure()
plt.bar(parametros[n], pesos_std, width = 0.5, color = 'grey', alpha = 0.5)
plt.xlabel('Parámetros')
plt.ylabel('Importancia %')
plt.title('Importancia parámetros, ' + modelos[n][:-4] + ', Derivada (STD)')
#con esta parte rotamos las etiquetas de la gráfica
plt.xticks(rotation = 45, ha='right')
#este comando hace que la figura ocupe la mayor parte del área total de la figura
plt.tight_layout() 

#graficamos el % importancia
plt.figure()
plt.bar(parametros[n], pesos_media, width = 0.5, color = 'grey', alpha = 0.5)
plt.xlabel('Parámetros')
plt.ylabel('Importancia %')
plt.title('Importancia parámetros, ' + modelos[n][:-4] + ', Derivada (MEDIA)')
#con esta parte rotamos las etiquetas de la gráfica
plt.xticks(rotation = 45, ha='right')
#este comando hace que la figura ocupe la mayor parte del área total de la figura
plt.tight_layout() 

#evaluamos la relevancia del parametro (input)
RI = RMSD_i/sum(RMSD_i)*100

#graficamos la relevancia (RI)
plt.figure()
plt.bar(parametros[n], RI.flatten(), width = 0.5, color = 'grey', alpha = 0.5)
plt.xlabel('Input Parameters')
plt.ylabel('Importance %')
plt.title('$RI_{PaD}$' + modelos[n][:-6] + ' 8')
#con esta parte rotamos las etiquetas de la gráfica
plt.xticks(rotation = 45, ha='right')
#este comando hace que la figura ocupe la mayor parte del área total de la figura
plt.tight_layout() 

###############################################################################
#        ESTIMACIÓN DE LA IMPORTANCIA POR EL ALGORITMO DE GARSON              #
###############################################################################

#separamos los valores de los pesos (pero solamente los de la 1ra capa)
#pesos = np.array(red.coefs_[0]).T
#OJO, cuando se usa .get_weigts el [0] da los valores de los pesos de la 1ra
#capa, [1] da los umbrales de la 1ra capa, [2] da los valores de los pesos de
#la 2da capa, [3] da los umbrales de la 2da capa... y así sucesivamente...
pesos = np.array(red.get_weights()[0])

#sumamos todos los pesos que le corresponde a cada parametro de entrada
pesos = sum(abs(pesos.T))

#evaluamos importancia% de cada parametro
importancia = pesos/sum(pesos)*100

#graficamos los niveles de importancia
plt.figure()
plt.bar(parametros[n], pesos, width = 0.5, color = 'grey', alpha = 0.5)
plt.xlabel('Input Parameters')
plt.ylabel('Importance %')
plt.title('Importancia parámetros, ' + modelos[n][:-5] + ' Método de Garson')
#con esta parte rotamos las etiquetas de la gráfica
plt.xticks(rotation = 45, ha='right')
#este comando hace que la figura ocupe la mayor parte del área total de la figura
plt.tight_layout() 

#generamos un vector para el acomodo de los datos en la grafica de barras
X = np.arange(len(parametros[n]))

#graficamos las dos series en un mismo gáfico de barras
plt.figure()
plt.bar(X + 0.5, pesos, width = 0.3, alpha = 0.5, label = 'Garson')
plt.bar(X + 0.2, RI.flatten(), width = 0.3, alpha = 0.5, label = 'PaD')
#plt.bar(X - 0.3, pesos_media, width = 0.3, alpha = 0.5, label = 'Der(MEDIA)')
plt.legend()
plt.xlabel('Input Parameters')
plt.ylabel('Importance %')
plt.title('RI')
plt.xticks([r + 0.3 for r in range(len(parametros[n]))], parametros[n])
#con esta parte rotamos las etiquetas de la gráfica
plt.xticks(rotation = 45, ha='right')
#este comando hace que la figura ocupe la mayor parte del área total de la figura
plt.tight_layout()
plt.savefig(ruta_figuras + 'RI_PaD_Garson.pdf', dpi = 300)

#analizamos el gráfico de violin para determinar si nuestro escalamiento fue 
#correcto
# plt.figure()
# sns.violinplot(x, saturation = 0.5)
# plt.xticks(range(len(parametros[n])), parametros[n])
# plt.xticks(rotation = 45, ha='right')
# plt.title('Parámetros escalados')
