# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 13:52:49 2023

@author: hecto
"""

#script para hacer optimización del número de capas para un modelo de red neuronal

#damos de alta las librerias que vamos a usar
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import time
from tabulate import tabulate
from scipy import stats
import seaborn as sns

#damos de alta la ruta donde estan los datos experimentales y el nombre del archivo
ruta_datos = 'C:/Users/hecto/OneDrive/Documentos/ITA/Posdoc Proyecto/articulo 1/codigos/base_datos/datos_unidos/datos_final.csv'
ruta_figuras = 'C:/Users/hecto/OneDrive/Documentos/ITA/Posdoc Proyecto/articulo 1/'

#leemos nuestra base de datos
df = pd.read_csv(ruta_datos)

#la ruta donde estan los maximos y minimos para escalar los datos
ruta_max_min = 'C:/Users/hecto/OneDrive/Documentos/ITA/Posdoc Proyecto/articulo 1/codigos/base_datos/datos_unidos/escalamiento.csv'

#leemos el dataframe de los valores maximos y minimos
df_max_min = pd.read_csv(ruta_max_min)

#cambiamos de nombre los encabezados del df
df.rename(columns = {'Hora':'Time', 'DV':'WD', 'HR':'RH', 'PM25':'PM2.5(t)', 
                     'VV':'WS', 'PM25_t':'PM2.5(t-1)'}, inplace = True)

#damos de alta los parametros de entada a la red y salida
parametros_entrada = ['Time', 'WeekD', 'YearD', 'TMP', 'WS', 'WD', 'RH', 'PP', 'PM2.5(t-1)']
parametros_salida = ['PM2.5(t)']

#separamos el dataframe y los maximos y minimos de cada parametro
df_entrada = df[parametros_entrada]
df_salida = df[parametros_salida]
df_max_min = df_max_min[parametros_entrada]

#eliminamos el df original
del(df)

#damos de alta la función para el escalamiento de los datos
def escal_maxmin(X,min_max):
    X_esc = (X-min_max[1,:])/(min_max[0,:]-min_max[1,:])
    return X_esc

#convertimos el df de los maximos y minimos en un arreglo numpy
min_max = np.array(df_max_min)

#convertimos el df de los parametros de entrada a un arreglo y los escalamos
x = np.array(df_entrada)
x = escal_maxmin(x, min_max)

#convertimos el df de salida a un arreglo
y = np.array(df_salida)

#eliminamos los dataframes que ya convertimos
del(df_entrada, df_max_min, df_salida)

#separamos los conjutos de datos en conjuntos de entrenamiento, validación y prueba
X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle = False)

#damo de alta el núm de capas a iterar (m) y el núm de neuronas (n)
m = 5
n = [10, 50, 100, 200, 300]

#generamos una clase para guadar la información de cada red entrenada
class modelo:
    def __init__(self, error, tiempo, capas, neuronas, r2):
        self.error = error
        self.tiempo = tiempo
        self.capas = capas
        self.neuronas = neuronas
        self.r2 = r2
    
    #esta función evalua el promedio de la lista de errores
    def prom_error(self):
        self.promedio_error = np.mean(np.array(self.error))
        return self.promedio_error
    
    #esta función evalua el promedio de la lista de tiempos
    def prom_time(self):
        self.promedio_time = np.mean(np.array(self.tiempo))
        return self.promedio_time
    
    #esta función evalua el promedio de la lista de r2
    def prom_r2(self):
        self.promedio_r2 = np.mean(np.array(self.r2))
        return self.promedio_r2
    
#generamos una lista vacia donde vamos a guardar la información de los modelos
modelos = []

#este ciclo va moviendo el número de neuronas ("n") por cada capa
for h in range(len(n)):
    #contruimos y entrenamos los modelos de red nerunal por cada capa "m"
    for i in range(m):
        #generamos una lista donde vamos a guardar el mse de cada entrenamiento y el tiempo
        error = []
        tiempos = []
        r2 = []
        
        #hacemos los entrenmamientos por triplicados
        for j in range(5):
            #generamos el modelo de red
            red = Sequential()
            
            #construcción del modelo segun el núm de capas
            for k in range(i+1):
                #agregamos las capas al modelo
                if k == 0:
                    red.add(Dense(n[h], input_dim = X_train.shape[1], activation = 'tanh'))
                    #red.add(LSTM(n, input_shape = (X_train.shape[1],24), activation = 'tanh', return_sequences=True))
                if k > 0 and k <= i:
                    red.add(Dense(n[h], activation = 'tanh'))
                    # if k <= i:
                    #     red.add(LSTM(n, activation = 'tanh'))
                    # else:
                    #     red.add(LSTM(n, activation = 'tanh', return_sequences=True))
                if k == i:
                    red.add(Dense(1, activation = 'linear'))
                
            #generamos el optimizador, agregamos al argumento los parametros del optimizador
            optimizador = SGD(learning_rate = 0.001)
            
            #compilamos el modelo de red
            red.compile(loss = 'mean_squared_error', optimizer = optimizador, metrics = ['mae','mse'])
            
            #comenzamos a medir el tiempo de entrenamiento
            inicio = time.time()
            
            #entrenamos el modelo
            red.fit(X_train, y_train, epochs = 100, validation_split = 0.1, verbose = False, shuffle = True)
            
            #paramos el tiempo
            fin = time.time()
            
            #guardamos el error y el tiempo
            error.append(red.history.history['loss'][-1])
            tiempos.append(fin-inicio)
            
            #graficamos la función de perdida
            # fig = plt.figure() 
            # plt.plot(red.history.history['loss'], 'k-')
            # plt.plot(red.history.history['val_loss'],'k--')
            # plt.title('Función de pérdida, ' + str(i+1) + ' capas')
            # plt.ylabel('Pérdida (MSE)')
            # plt.xlabel('Epocas')
            # plt.legend(['Entrenamiento', 'Validación'], loc='upper right')
            # plt.show()
            
            #evaluamos la red y guardamos su valor de r2
            #OJO!! ESTO TIENE QUE SIMPRE DESPUES DE CONSULTAR LAS FUNCIONES DE
            #PERDIDA DEL MODELO DE RED, PORQUE SI SE HACE ANTES SE PIERDE LA INFORMACIÓN
            #Y YA NO SE PUEDE ACCEDER A ['loss']
            r2.append(r2_score(y_train, red.predict(X_train)))
            
            #graficamos la evaluación de modelo contra la info real y obtenemos el coeficiente de determinación
            # plt.figure()
            # plt.plot(red.predict(X_train), y_train, 'ko', markerfacecolor = 'None')
            # plt.plot([0,max(y_train)],[0,max(y_train)],'--', color = 'red')
            # plt.xlabel('Modelo')
            # plt.ylabel('Real')
            # plt.title('Coeficiente de DETERMINACIÓN, $r^2 = $' + str(round(r2_score( y_train, red.predict(X_train)),4)))
            
            #madanmos a pantalla la estructura de la red
            red.summary()
        
        #guardamos los resultados del modelo
        modelos.append(modelo(error, tiempos, i+1, n[h], r2))

#generamos las listas vacias otra vez
# error = []
# tiempos = []
# capas = []
# r2 = []

# #guardamos cada valor 
# for i in range(len(modelos)):
#     error.append(modelos[i].prom_error())
#     tiempos.append(modelos[i].prom_time())
#     capas.append(modelos[i].capas)
#     r2.append(modelos[i].prom_r2())
    
#graficamos el error (MSE) y la r2
# fig, ax1 = plt.subplots()
# ax1.plot(capas, error, '-o', markerfacecolor = 'white', color = 'blue', label = 'MSE')
# ax1.set_ylabel('Error (MSE)')
# ax1.set_xlabel('Capas')

# #creamos el segundo eje
# ax2 = ax1.twinx()
# ax2.plot(capas, r2, '-s', markerfacecolor = 'white', color = 'red', label = '$r^{\ 2}$')
# ax2.set_ylabel('$r^{\ 2}$')
# plt.title('MSE y $r^{\ 2}$ del entrenamiento de cada modelo, ' + str(n) + ' neuronas')

# #para agregar las leyendas se hace lo siguiente
# linea1, leyenda1 = ax1.get_legend_handles_labels()
# linea2, leyenda2 = ax2.get_legend_handles_labels()
# plt.legend(linea1 + linea2, leyenda1 + leyenda2)

# #graficamos el tiempo de computo (seg)
# plt.figure()
# plt.plot(capas, tiempos, '-o', markerfacecolor='white', color = 'black')
# plt.xlabel('Capas')
# plt.ylabel('Tiempo (seg)')
# plt.title('Tiempo de computo en el entrenamiento del modelo')

#generamos un df donde guadraremos la info de cada entrenamiento
df_info = pd.DataFrame(columns = ['neuronas'] + [str(j) + '_' + str(i) for i in range(m) for j in ['err','r2','time']])
#info = np.zeros([5,3*len(modelos)])

#generamos las dimensiones del df
df_info['neuronas'] = np.array([np.nan]*len(n))

#contador para ir generando un avance
nn = 0

#generamos la matriz de información
for i in range(len(n)):
    #contador para las posiciones
    mm = 1
    for j in range(m):
        df_info.iat[i,0] = modelos[nn].neuronas
        df_info.iat[i,mm] = modelos[nn].prom_error()
        df_info.iat[i,mm + 1] = modelos[nn].prom_r2()
        df_info.iat[i,mm + 2] = modelos[nn].prom_time()
        # info[0:3,n] = modelos[i].error
        # info[0:3,n+1] = modelos[i].r2
        # info[0:3,n+2] = modelos[i].tiempo
        # info[3:5,n] = [np.mean(modelos[i].error), np.std(modelos[i].error, ddof = 1)]
        # info[3:5,n+1] = [np.mean(modelos[i].r2), np.std(modelos[i].r2, ddof = 1)]
        # info[3:5,n+2] = [np.mean(modelos[i].tiempo), np.std(modelos[i].tiempo, ddof = 1)]
        nn = nn + 1
        mm = mm + 3

#convertimos todos los elementos del df en flotantes
df_info = df_info.astype(float)

#generamos nuestras X (capas) y Y (neuronas) para los ejes de las graficas de superficie
Y = np.array(df_info.neuronas)
X = np.arange(m) + 1

#generamos las matrices
X, Y = np.meshgrid(X,Y)

#sacamos las información de la matriz de datos
#para esto generamos tres matrices diferentes con la información
Z_MSE, Z_r2, Z_tiempo = np.zeros(X.shape), np.zeros(X.shape), np.zeros(X.shape)

#generamos un par de contadores
cont1 = 0

#generamos un par de ciclos para ir sacando la información
# for i in range(X.shape[1]):
#     for j in range(X.shape[0]):
#         Z_MSE[j,i] = datos[m,n]
#         Z_r2[j,i] = datos[m,n+1]
#         Z_tiempo[j,i] = datos[m,n+2]
#         m = m + 5
#     n = n + 3
#     m = 5

#en este ciclo sacamos la info
for i in range(m):
    Z_MSE[:,i] = df_info.iloc[:,1 + cont1]
    Z_r2[:,i] = df_info.iloc[:,2 + cont1]
    Z_tiempo[:,i] = df_info.iloc[:,3 + cont1]
    cont1 = cont1 + 3    

#graficamos las superficies de los resultados obtenidos
fig = plt.figure(figsize=(10,6))
ax = plt.axes(projection = '3d')

surf1 = ax.plot_surface(X, Y, Z_MSE, cmap = 'viridis')
ax.set_xlabel('Capas')
ax.set_ylabel('Neuronas')
ax.set_zlabel('MSE', rotation = -90)
ax.set_title('Error medio cudratico (MSE)')
ax.zaxis.labelpad=-0.5
fig.colorbar(surf1, ax=ax, shrink=0.5, aspect=5)
plt.show()


fig = plt.figure(figsize=(10,6))
ax = plt.axes(projection = '3d')

surf1 = ax.plot_surface(X, Y, Z_r2, cmap = 'viridis')
ax.set_xlabel('Capas')
ax.set_ylabel('Neuronas')
ax.set_zlabel('$r^2$', rotation = -90)
ax.set_title('Coeficiente de determinación ($r^2$)')
fig.colorbar(surf1, ax=ax, shrink=0.5, aspect=5)
plt.show()

fig = plt.figure(figsize=(10,6))
ax = plt.axes(projection = '3d')

ax.plot_surface(X, Y, Z_tiempo, cmap = 'viridis')
ax.set_xlabel('Capas')
ax.set_ylabel('Neuronas')
ax.set_zlabel('Tiempo (s)')
ax.set_title('Tiempo de entrenamiento')
fig.colorbar(surf1, ax=ax, shrink=0.5, aspect=5)
plt.show()

#generamos los graficos de contorno de los mismos resultados
plt.contourf(X,Y,Z_MSE, levels = 100, vmax = Z_MSE.max(), vmin = Z_MSE.min(), cmap ='turbo')
plt.plot([1], [50], 'ko', markerfacecolor = 'white', markersize = 10, label = 'Model_8') #estructura de red utilizada
plt.plot(X[np.where(Z_MSE == Z_MSE.min())], Y[np.where(Z_MSE == Z_MSE.min())], 'ks', markerfacecolor = 'white', markersize = 10, label = 'Best') #estructura de red optima
plt.legend()
plt.xlabel('Layers')
plt.ylabel('Neurons')
plt.colorbar().set_label('MSE')
plt.title('Mean squared error')
plt.xticks(np.arange(1, m + 1 , step=1))
plt.savefig(ruta_figuras + 'MSE_topologia.pdf', dpi = 300)
plt.show()

plt.contourf(X,Y,Z_r2, levels = 100, vmax = Z_r2.max(), vmin = Z_r2.min(), cmap ='turbo')
plt.plot([1], [50], 'ko', markerfacecolor = 'white', markersize = 10, label = 'Model_8') #estructura de red utilizada
plt.plot(X[np.where(Z_r2 == Z_r2.max())], Y[np.where(Z_r2 == Z_r2.max())], 'ks', markerfacecolor = 'white', markersize = 10, label = 'Best') #estructura de red optima
plt.legend()
plt.xlabel('Layers')
plt.ylabel('Neurons')
plt.colorbar().set_label('$R^2$')
plt.title('Coefficient of determination')
plt.xticks(np.arange(1, m + 1 , step=1))
plt.savefig(ruta_figuras + 'r2_topologia.pdf', dpi = 300)
plt.show()

plt.contourf(X,Y,Z_tiempo, levels = 100, vmax = Z_tiempo.max(), vmin = Z_tiempo.min(), cmap = 'turbo')
plt.plot([1], [50], 'ko', markerfacecolor = 'white', markersize = 10, label = 'Model_8') #estructura de red utilizada
plt.plot(X[np.where(Z_tiempo == Z_tiempo.min())], Y[np.where(Z_tiempo == Z_tiempo.min())], 'ks', markerfacecolor = 'white', markersize = 10, label = 'Best') #estructura de red optima
plt.legend()
plt.xlabel('Layers')
plt.ylabel('Neurons')
plt.colorbar().set_label('Time (s)')
plt.title('Training time')
plt.xticks(np.arange(1, m + 1 , step=1))
plt.savefig(ruta_figuras + 'time_topologia.pdf', dpi = 300)
plt.show()

#generamos una tabla con las desviaciones estandar del error (MSE)
df_error = pd.DataFrame(np.full([m,len(n)], np.nan), columns = [str(i) for i in range(1,6)])

#contador para los modelos
conta2 = 0
error = np.zeros([5,2])
r2 = np.zeros([5,2])
nn = 0
mm = 0

#separamos los datos
for i in range(len(n)):
    for j in range(m):
        df_error.iat[len(n) - i - 1, j] = np.std(modelos[conta2].error)
        if modelos[conta2].capas == 1 and modelos[conta2].neuronas == 50:
            r2[:,nn] = modelos[conta2].r2
            error[:,mm] = modelos[conta2].error
            nn = nn + 1
            mm = mm + 1
        if modelos[conta2].capas == X[np.where(Z_r2 == Z_r2.max())] and modelos[conta2].neuronas == Y[np.where(Z_r2 == Z_r2.max())]:
            r2[:,nn] = modelos[conta2].r2
            nn = nn + 1
        if modelos[conta2].capas == X[np.where(Z_MSE == Z_MSE.min())] and modelos[conta2].neuronas == Y[np.where(Z_MSE == Z_MSE.min())]:
            error[:,mm] = modelos[conta2].error
            mm = mm + 1
        conta2 = conta2 + 1
    
plt.boxplot(error, labels= ['9-50-1'] + ['9-' + ''.join(str(Y[np.where(Z_MSE == Z_MSE.min())][0])[:-2] + '-' for i in range(X[np.where(Z_MSE == Z_MSE.min())][0])) + '1'],
            showmeans = True, meanline = True, widths = 0.5)
plt.ylabel('MSE')
plt.xlabel('ANN Topology')
plt.title('t-test, p-value = ' + str(round(stats.ttest_ind(error[:,0], error[:,1])[1],6)))
plt.grid()
plt.savefig(ruta_figuras + 'boxplot_MSE_topo.pdf', dpi = 300)
plt.show()

plt.boxplot(r2, labels= ['9-50-1'] + ['9-' + ''.join(str(Y[np.where(Z_r2 == Z_r2.max())][0])[:-2] + '-' for i in range(X[np.where(Z_r2 == Z_r2.max())][0])) + '1'],
            showmeans = True, meanline = True, widths = 0.5)
plt.ylabel('$R^2$')
plt.xlabel('ANN Topology')
plt.title('t-test, p-value = ' + str(round(stats.ttest_ind(r2[:,0], r2[:,1])[1],6)))
plt.grid()
plt.savefig(ruta_figuras + 'boxplot_r2_topo.pdf', dpi = 300)
plt.show()


#concatenamos
df_neuronas = pd.DataFrame(np.flip(np.array(n)), columns=['Neurons'])
df_error = pd.concat([df_neuronas, df_error], axis = 1)
df_error = round(df_error, 4)

#generamos la tabla para el archivo de latex
tabla1 = tabulate(df_error, tablefmt='latex', headers='keys', showindex=False)
