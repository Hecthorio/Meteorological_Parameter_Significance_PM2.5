# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 14:32:43 2023

@author: hecto
"""

#Script para entrenar los multiples modelos utiizando diferentes parámetros
#de entrada

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD, RMSprop
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.stats import kurtosis, skew
from tabulate import tabulate
from keras.models import load_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf
from scipy.stats import shapiro, kstest, norm
import seaborn as sns

#1ero vamos a leer la base de datos y haremos el escalamiento de los mismos

#datos de alta la ruta donde esta la base de datos filtrada y el nombre del archivo
ruta = 'C:/Users/hecto/OneDrive/Documentos/ITA/Posdoc Proyecto/articulo 1/codigos/base_datos/datos_unidos/'
nombre_archivo = 'datos_final.csv'

#damos el nombre donde se van a ir guardando los modelos generados
ruta_modelos = 'C:/Users/hecto/OneDrive/Documentos/ITA/Posdoc Proyecto/articulo 1/codigos/modelos/'

#damos de alta la ruta donde se guardan las figuras
ruta_fig = 'C:/Users/hecto/OneDrive/Documentos/ITA/Posdoc Proyecto/articulo 1/'

#leemos el archivo
df = pd.read_csv(ruta + nombre_archivo)

#cambiamos de nombre los encabezados del df
df.rename(columns = {'Hora':'Time', 'DV':'WD', 'HR':'RH', 'PM25':'PM2.5(t)', 
                     'VV':'WS', 'PM25_t':'PM2.5(t-1)'}, inplace = True)

#vamos a sacar los maximos y minimos de cada parametro en el df para genera un 
#df quer servira para hacer el escalado de los datos
#función de escalamiento por máximos y mínimos para el entrenamiento de la red
def max_min (df,ruta):
    df_n = df.agg(['max','min'])
    df_n.to_csv(ruta + 'escalamiento.csv', index=True, header=True)
    return df_n

def escal_maxmin(X,min_max):
    X_esc = (X-min_max[1,:])/(min_max[0,:]-min_max[1,:])
    return X_esc

#generamos el archivo que contine los maximos y minimos
max_min(df,ruta)

#generamos una lista con los parametros de entrada que se utilizan paea cada modelo
modelos = {1:['Time', 'WeekD', 'YearD'],
            2:['Time', 'WeekD', 'YearD', 'TMP'],
            3:['Time', 'WeekD', 'YearD', 'WS'],
            4:['Time', 'WeekD', 'YearD', 'WD'],
            5:['Time', 'WeekD', 'YearD', 'RH'],
            6:['Time', 'WeekD', 'YearD', 'PP'],
            7:['Time', 'WeekD', 'YearD', 'PM2.5(t-1)'],
            8:['Time', 'WeekD', 'YearD', 'TMP', 'WS', 'WD', 'RH', 'PP', 'PM2.5(t-1)']}

#generamos unas listas para guardar información
perdida_train = []
perdida_val = []
mae_train = []
mse_train = []
mae_test = []
mse_test = []
r2_train = []
r2_test = []
y_mean_train = []
y_mean_test = []
y_std_train = []
y_std_test = []
y_obli_train = []
y_obli_test = []
y_kur_train = []
y_kur_test = []
resid_train = []
resid_test = []

#escalamos nuestro df
#OJO, para usar la función se tiene que usar arreglos np y no pd
#primero separamos el conjunto de datos que se va a emplear para el entrenamiento
#con base a el numero de modelo se va a entrenar (y tambien la salida)
#aqui tomamos el 8 como referencia porque tiene a todos los parámetros
X = df[modelos[8]].to_numpy()
y = df['PM2.5(t)'].to_numpy()

#leemos y separamos los max y min que vamos a utilizar
max_min = pd.read_csv(ruta + 'escalamiento.csv')

#lo convertimos a un arreglo np
max_min = max_min[modelos[8]].to_numpy()

#escalamos los parametros
X = escal_maxmin(X, max_min)

# max_min = pd.read_csv(ruta + 'escalamiento.csv')
# max_min = max_min['PM2.5(t)'].to_numpy()
# max_min = np.reshape(max_min, (len(max_min),1))
# y = np.reshape(y, (len(y),1))
# y = escal_maxmin(y, max_min)

#import scipy as sp
#transformed_data, lambda_value_y = sp.stats.boxcox(y)
#y = transformed_data
#X[:,8] = X[:,8]+1
#transformed_data_x, lambda_value_x = sp.stats.boxcox(X[:,8])
# X[:,8] = transformed_data_x

#separamos los datos para el entrenamiento y validación
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.9, shuffle = False)

#hacemos un reshape a las salidas
y_train = np.reshape(y_train, (len(y_train),1))
y_test = np.reshape(y_test, (len(y_test),1))

#evaluamos y guardamos la media, desviación estandar, oblicuidad y curtosis
#de los datos originales (observaciones)
#media:
y_mean_train.append(np.mean(y_train))
y_mean_test.append(np.mean(y_test))

#desviación estandar
y_std_train.append(np.std(y_train))
y_std_test.append(np.std(y_test))

#oblicuidad
y_obli_train.append(skew(y_train)[0])
y_obli_test.append(skew(y_test)[0])

#curtosis
y_kur_train.append(kurtosis(y_train)[0])
y_kur_test.append(kurtosis(y_test)[0])

#guardamos las variables de X_train y X_test para poder modificar entre cada iteración
X_train_save = pd.DataFrame(X_train, columns=modelos[8])
X_test_save = pd.DataFrame(X_test, columns=modelos[8])

#iniciamos el ciclo para ir entrenando los modelos y guardarlos
for i in range(len(modelos)):
    
    #separamos el conjunto de datos que vamos a utilizar para el entrenamiento
    X_train = X_train_save[modelos[i+1]].to_numpy()
    X_test = X_test_save[modelos[i+1]].to_numpy()
    
    #entrenamos los modelos de redes neuronales
    red = Sequential()
    red.add(Dense(50, input_dim=X_train.shape[1], activation = 'tanh'))
    #red.add(Dense(100, activation = 'tanh'))
    #OJO, 'selu', 'linear'
    red.add(Dense(1, activation='linear'))
    
    #Compilar el modelo
    optimizer = SGD(learning_rate = 0.001)
    red.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae', 'mse'])
    
    #entrenamiento del modelo de red
    red.fit(X_train, y_train, epochs = 150, validation_split = 0.1, shuffle = True)
    
    #guardamos la información de las funciones perdida de los datos de entrenamiento y validación
    perdida_train.append(red.history.history['loss'])
    perdida_val.append(red.history.history['val_loss'])
    
    # Evaluación del mae y mse entrenamiento y prueba y guardamos info
    eva_train = red.evaluate(X_train, y_train)
    mae_train.append(eva_train[1])
    mse_train.append(eva_train[2])
    eva_test = red.evaluate(X_test, y_test)
    mae_test.append(eva_test[1])
    mse_test.append(eva_test[2])
    
    #evaluamos la red con los datos de entrenamiento y prueba
    y_train_eval = red.predict(X_train)   #evaluar la red obtenida con todos los datos (entrenamiento y prueba)
    y_test_eval = red.predict(X_test)   #evaluar la red obtenida con todos los datos (entrenamiento y prueba)
    
    #evaluamos el coeficiente de determinación para el conjunto de datos de entrenamiento y prueba
    r2_train.append(metrics.r2_score(y_train,y_train_eval))
    r2_test.append(metrics.r2_score(y_test,y_test_eval))
    
    #evaluamos y guardamos la media, desviación estandar, oblicuidad y curtosis
    #media:
    y_mean_train.append(np.mean(y_train_eval))
    y_mean_test.append(np.mean(y_test_eval))
    
    #desviación estandar
    y_std_train.append(np.std(y_train_eval))
    y_std_test.append(np.std(y_test_eval))
    
    #oblicuidad
    y_obli_train.append(skew(y_train_eval)[0])
    y_obli_test.append(skew(y_test_eval)[0])
    
    #curtosis
    y_kur_train.append(kurtosis(y_train_eval)[0])
    y_kur_test.append(kurtosis(y_test_eval)[0])
    
    #guardamos los residuales de cada modelo
    resid_train.append(y_train - y_train_eval)
    resid_test.append(y_test - y_test_eval)
    
    #guardar los modelos
    red.save(ruta_modelos + 'modelo_' + str(i+1) + '.h5')

#guardamos los conjuntos de datos usados para el entrenamiento y validación
#por si los necesitamos después para hacer una evaluación
a = X_train_save.assign(PM = np.nan)
a['PM'] = y_train
a = a.rename(columns = {'PM':'PM2.5(t)'})
a.to_csv(ruta + 'train.csv', index = False)

a = X_test_save.assign(PM = np.nan)
a['PM'] = y_test
a = a.rename(columns = {'PM':'PM2.5(t)'})
a.to_csv(ruta + 'test.csv', index = False)

del(a)

#graficamos las funciones de perdida
for i in range(int(len(perdida_train)/2)):
    plt.plot(perdida_train[i], '-', label = 'Model_' + str(i+1))
plt.xlabel('Epocs')
plt.ylabel('MSE')
plt.title('Training')
plt.legend()
plt.savefig(ruta_fig + 'loss_fun_train1.pdf',dpi = 300)
plt.show()

for i in range(int(len(perdida_train)/2)):
    plt.plot(perdida_train[i+4], '-', label = 'Model_' + str(i+5))
plt.xlabel('Epocs')
plt.ylabel('MSE')
plt.title('Training')
plt.legend()
plt.savefig(ruta_fig + 'loss_fun_train2.pdf',dpi = 300)
plt.show()

for i in range(int(len(perdida_val)/2)):
    plt.plot(perdida_val[i], '-', label = 'Model_' + str(i+1))
plt.xlabel('Epocs')
plt.ylabel('MSE')
plt.title('Test')
plt.legend()
plt.savefig(ruta_fig + 'loss_fun_test1.pdf',dpi = 300)
plt.show()

for i in range(int(len(perdida_val)/2)):
    plt.plot(perdida_val[i+4], '-', label = 'Model_' + str(i+5))
plt.xlabel('Epocs')
plt.ylabel('MSE')
plt.title('Test')
plt.legend()
plt.savefig(ruta_fig + 'loss_fun_test2.pdf',dpi = 300)
plt.show()

#generamos las tablas que contendran las medias, desviación estandar, oblicuidad y curtosis
#para eso vamos a generar un df con esa información
df_estad = pd.DataFrame(columns=['Model','Mean','SD','Skewness','Kurtosis'])
df_estad['Model'] = ['Observed','1','2','3','4','5','6','7','8']
df_estad['Mean'] = y_mean_train
df_estad['SD'] = y_std_train
df_estad['Skewness'] = y_obli_train
df_estad['Kurtosis'] = y_kur_train
df_estad = round(df_estad,4)

#generamos la tabla de latex
tabla = tabulate(df_estad, tablefmt='latex', headers='keys', showindex=False)

df_estad_test = pd.DataFrame(columns=['Model','Mean','SD','Skewness','Kurtosis'])
df_estad_test['Model'] = ['Observed','1','2','3','4','5','6','7','8']
df_estad_test['Mean'] = y_mean_test
df_estad_test['SD'] = y_std_test
df_estad_test['Skewness'] = y_obli_test
df_estad_test['Kurtosis'] = y_kur_test
df_estad_test = round(df_estad_test,4)

#generamos la tabla de latex
tabla2 = tabulate(df_estad_test, tablefmt='latex', headers='keys', showindex=False)

df_metric = pd.DataFrame(columns = ['Model', 'MSE_train', 'MAE_train', 'R^2_train', 'MSE_test', 'MAE_test', 'R^2_test'])
df_metric['Model'] = ['1','2','3','4','5','6','7','8']
df_metric['MSE_train'] = mse_train
df_metric['MAE_train'] = mae_train
df_metric['R^2_train'] = r2_train
df_metric['MSE_test'] = mse_test
df_metric['MAE_test'] = mae_test
df_metric['R^2_test'] = r2_test
df_metric = round(df_metric,4)

#generamos la tabla de latex
tabla3 = tabulate(df_metric, tablefmt='latex', headers='keys', showindex=False)

#comparación de respuesta del último modelo con los datos de entrenamiento y prueba
plt.figure()
plt.plot(y_train, y_train_eval, 'ko', markerfacecolor = 'None')
plt.plot([-5,85],[-5,85],'r--')
plt.xlabel('Observed ($c_i$)')
plt.ylabel('Predicted ($y_{ann}$)')
plt.title('Train')
plt.ylim(-5,85)
plt.xlim(-5,85)
plt.legend(['Model 8', '$y_{ann}=c_{i}$'])
plt.savefig(ruta_fig + 'pred_obser_train.pdf', dpi = 300)

plt.figure()
plt.plot(y_test, y_test_eval, 'ko', markerfacecolor = 'None')
plt.plot([-5,85],[-5,85],'r--')
plt.xlabel('Observed ($c_i$)')
plt.ylabel('Predicted ($y_{ann}$)')
plt.title('Test')
plt.ylim(-5,85)
plt.xlim(-5,85)
plt.legend(['Model 8', '$y_{ann}=c_{i}$'])
plt.savefig(ruta_fig + 'pred_obser_test.pdf', dpi = 300)

#de aquí en adelante se hace el analsis de los residuales del modelo
#ahora vamos a leer los archivos donde estan los datos de entrenamiento y validación
#porque vamos a ordenar los datos para evaluar el modelos y obtener los residuales ordenados
#para despues aplicar las funciones de ACF y PACF
a_train = pd.read_csv(ruta + 'train.csv')
a_test = pd.read_csv(ruta + 'test.csv')

#ordenamos el df para poder despues de evaluar poder usar ACF y PACF
a_train = a_train.sort_values(by=['YearD','Time'])
a_test = a_test.sort_values(by=['YearD','Time'])

#cargamos el modelo
modelo1 = load_model(ruta_modelos + 'modelo_1.h5')
modelo8 = load_model(ruta_modelos + 'modelo_8.h5')

#en este ciclo evaluamos las ACF y PACF para el peor y mejor modelo
for j in range(2):
    #en esta sección intercambiamos entre datos de entrenamiento y validación
    if j == 0:
        a = a_train
    else:
        a = a_test
    for i in range(2):
        #en esta sección cambiamos entre los modelos
        if i == 0:
            modelo = modelo1
            b = modelo.predict(a[modelos[1]].to_numpy())
        else:
            modelo = modelo8
            b = modelo.predict(a[modelos[8]].to_numpy())
        
        resi = a['PM2.5(t)'].to_numpy() - b.T
    
        # Graficar la Función de Autocorrelación (ACF)
        plot_acf(resi.T, lags=23)
        plt.title('Auto Correlation Function')
        plt.xlabel('Lag')
        plt.ylabel('ACF')
        plt.legend(['Model 1' if i == 0 else 'Model 8'])
        plt.savefig(ruta_fig + 'ACF_modelo' + ('1' if i == 0 else '8') + ('train' if j == 0  else 'test') + '.pdf', dpi = 300)
        plt.show()
        
        # Graficar la Función de Autocorrelación Parcial (PACF)
        plot_pacf(resi.T, lags=23)
        plt.title('Partial Autocorrelation Function')
        plt.xlabel('Lag')
        plt.ylabel('PACF')
        plt.legend(['Model 1' if i == 0 else 'Model 8'])
        plt.savefig(ruta_fig + 'PACF_modelo' + ('1' if i == 0 else '8') + ('train' if j == 0  else 'test') + '.pdf', dpi = 300)
        plt.show()

###############################################################################
#             TABLAS DE MEDIAS, HOMOCEDASTICIDAD, NORMALIDAD Y                #
#                    ACF DE LOS RESIDUALES DE TRAIN Y TEST                    #
###############################################################################

#generamos un df con las medias de los residuales, la prueba de breanch-pagan (p-valor), Shapiro-Test y lag1 de los residuales
resid_df_train = pd.DataFrame(columns = ['Model','e','BP','KS','rho_1'])
resid_df_test = pd.DataFrame(columns = ['Model','e','BP','KS','rho_1'])
resid_df_train['Model'] = ['1','2','3','4','5','6','7','8']
resid_df_test['Model'] = ['1','2','3','4','5','6','7','8']
for i in range(8):
    #evaluamos la media de los residuales
    resid_df_train.iloc[i,1] = format(np.mean(resid_train[i]),'.4g')
    resid_df_test.iloc[i,1] = format(np.mean(resid_test[i]),'.4g')
    #en esta parte reacomodamos los errores usando los indices de los dataframes
    #ordenados para poder evaluar ACP del error
    resid_model = a_train.join(pd.DataFrame(resid_train[i].flatten()))
    resid_model = resid_model[0]
    resid_df_train.iloc[i,4] = format(acf(resid_model, nlags = 1)[1],'.4g')
    resid_model = a_test.join(pd.DataFrame(resid_test[i].flatten()))
    resid_model = resid_model[0]
    resid_df_test.iloc[i,4] = format(acf(resid_model, nlags = 1)[1],'.4g')
    #evaluamos el p-valor para la prueba heterocedasticidad
    resid_std = (resid_train[i] - np.mean(resid_train[i]))/np.std(resid_train[i])
    resid_df_train.iloc[i,2] = format(sm.stats.diagnostic.het_breuschpagan(resid_std.flatten(), X_train_save[modelos[i+1]].to_numpy())[3],'.4g')
    resid_std = (resid_test[i] - np.mean(resid_test[i]))/np.std(resid_test[i])
    resid_df_test.iloc[i,2] = format(sm.stats.diagnostic.het_breuschpagan(resid_std.flatten(), X_test_save[modelos[i+1]].to_numpy())[3],'.4g')
    #evaluamos la normalidad del modelo con la prueba de Kolmogorov-Smirnov
    resid_std = (resid_train[i] - np.mean(resid_train[i]))/np.std(resid_train[i])
    resid_df_train.iloc[i,3] = format(kstest(resid_std.flatten(), 'norm')[1],'.4g')
    resid_std = (resid_test[i] - np.mean(resid_test[i]))/np.std(resid_test[i])
    resid_df_test.iloc[i,3] = format(kstest(resid_std.flatten(), 'norm')[1],'.4g')
    #convertimos los str a float
    # resid_df_train = resid_df_train.astype(float)
    # resid_df_test = resid_df_test.astype(float)
    # resid_df_train['Model'] = resid_df_train['Model'].astype(int)
    # resid_df_test['Model'] = resid_df_test['Model'].astype(int)
    
    
#creamos las tablas que vamos a exportar al articulo
tabla4 = tabulate(resid_df_train, tablefmt='latex', headers='keys', showindex=False)
tabla5 = tabulate(resid_df_test, tablefmt='latex', headers='keys', showindex=False)

###############################################################################
#                       HISTOGRAMAS RESIDUALES                                #
###############################################################################

#Graficamos los histogramas de los residuales del 1er modelo y 8vo modelo de junto con
#el grafico de los cuartiles
sns.histplot(resid_train[0], alpha = 0, stat='density', bins = 20)
sns.kdeplot(resid_train[0], color= 'k', linestyle="--")
plt.xlabel('Residual')
plt.legend(['Model 1'])
plt.savefig(ruta_fig + 'hist_modelo1_train.pdf', dpi = 300)
plt.show()

sns.histplot(resid_test[0], alpha = 0, stat='density', bins = 20)
sns.kdeplot(resid_test[0], color= 'k', linestyle="--")
plt.xlabel('Residual')
plt.legend(['Model 1'])
plt.savefig(ruta_fig + 'hist_modelo1_test.pdf', dpi = 300)
plt.show()

sns.histplot(resid_train[7], alpha = 0, stat='density', bins = 20)
sns.kdeplot(resid_train[7], color= 'k', linestyle="--")
plt.xlabel('Residual')
plt.legend(['Model 8'])
plt.savefig(ruta_fig + 'hist_modelo8_train.pdf', dpi = 300)
plt.show()

sns.histplot(resid_test[7], alpha = 0, stat='density', bins = 20)
sns.kdeplot(resid_test[7], color= 'k', linestyle="--")
plt.xlabel('Residual')
plt.legend(['Model 8'])
plt.savefig(ruta_fig + 'hist_modelo8_test.pdf', dpi = 300)
plt.show()

##############################################################################
#                       GRAFICAS QQ-PLOT RESIDUALES                          #
##############################################################################

#estandarizamos los residuales para generar las graficas q-q plot
resid_std = (resid_train[0] - np.mean(resid_train[0]))/np.std(resid_train[0])
pp = sm.ProbPlot(resid_std.flatten())
qq = pp.qqplot(marker='o', markerfacecolor='None', markeredgecolor='k')
sm.qqline(qq.axes[0], line='45', fmt='r-', linestyle='--')
plt.ylabel('Standardized Residual')
plt.xlabel('Standard Normal Quantiles')
plt.legend(['Model 1'])
plt.savefig(ruta_fig + 'qqplot_modelo1_train.pdf', dpi = 300)
plt.show()

resid_std = (resid_train[7] - np.mean(resid_train[7]))/np.std(resid_train[7])
pp = sm.ProbPlot(resid_std.flatten())
qq = pp.qqplot(marker='o', markerfacecolor='None', markeredgecolor='k')
sm.qqline(qq.axes[0], line='45', fmt='r-', linestyle='--')
plt.ylabel('Standardized Residual')
plt.xlabel('Standard Normal Quantiles')
plt.legend(['Model 8'])
plt.savefig(ruta_fig + 'qqplot_modelo8_train.pdf', dpi = 300)
plt.show()

resid_std = (resid_test[0] - np.mean(resid_test[0]))/np.std(resid_test[0])
pp = sm.ProbPlot(resid_std.flatten())
qq = pp.qqplot(marker='o', markerfacecolor='None', markeredgecolor='k')
sm.qqline(qq.axes[0], line='45', fmt='r-', linestyle='--')
plt.ylabel('Standardized Residual')
plt.xlabel('Standard Normal Quantiles')
plt.legend(['Model 1'])
plt.savefig(ruta_fig + 'qqplot_modelo1_test.pdf', dpi = 300)
plt.show()

resid_std = (resid_test[7] - np.mean(resid_test[7]))/np.std(resid_test[7])
pp = sm.ProbPlot(resid_std.flatten())
qq = pp.qqplot(marker='o', markerfacecolor='None', markeredgecolor='k')
sm.qqline(qq.axes[0], line='45', fmt='r-', linestyle='--')
plt.ylabel('Standardized Residual')
plt.xlabel('Standard Normal Quantiles')
plt.legend(['Model 8'])
plt.savefig(ruta_fig + 'qqplot_modelo8_test.pdf', dpi = 300)
plt.show()

##############################################################################
#           GRAFICAS DE RESIDUALES VS VALOR PREDICHO/TIEMPO                  #
##############################################################################

#graficamos los residuales contra el valor los valores predichos y agregamos
#tambien las lineas de los intervalos de confianza
resid_std = (resid_train[0] - np.mean(resid_train[0]))/np.std(resid_train[0])
plt.plot(y_train,resid_std,'ko', markerfacecolor = 'None')
plt.axhline(1.96, color = 'b', linestyle='--')
plt.axhline(0, color = 'r', linestyle='--')
plt.axhline(-1.96, color = 'b', linestyle='--')
plt.legend(['Model 1'])
plt.xlabel('Predicted Value')
plt.ylabel('Standardized Residual')
plt.savefig(ruta_fig + 'resid_predic_modelo1_train.pdf', dpi = 300)
plt.show()

resid_std = (resid_train[7] - np.mean(resid_train[7]))/np.std(resid_train[7])
plt.plot(y_train,resid_std,'ko', markerfacecolor = 'None')
plt.axhline(1.96, color = 'b', linestyle='--')
plt.axhline(0, color = 'r', linestyle='--')
plt.axhline(-1.96, color = 'b', linestyle='--')
plt.legend(['Model 8'])
plt.xlabel('Predicted Value')
plt.ylabel('Standardized Residual')
plt.savefig(ruta_fig + 'resid_predic_modelo8_train.pdf', dpi = 300)
plt.show()

resid_std = (resid_test[0] - np.mean(resid_test[0]))/np.std(resid_test[0])
plt.plot(y_test,resid_std,'ko', markerfacecolor = 'None')
plt.axhline(1.96, color = 'b', linestyle='--')
plt.axhline(0, color = 'r', linestyle='--')
plt.axhline(-1.96, color = 'b', linestyle='--')
plt.legend(['Model 1'])
plt.xlabel('Predicted Value')
plt.ylabel('Standardized Residual')
plt.savefig(ruta_fig + 'resid_predic_modelo1_test.pdf', dpi = 300)
plt.show()

resid_std = (resid_test[7] - np.mean(resid_test[7]))/np.std(resid_test[7])
plt.plot(y_test,resid_std,'ko', markerfacecolor = 'None')
plt.axhline(1.96, color = 'b', linestyle='--')
plt.axhline(0, color = 'r', linestyle='--')
plt.axhline(-1.96, color = 'b', linestyle='--')
plt.legend(['Model 8'])
plt.xlabel('Predicted Value')
plt.ylabel('Standardized Residual')
plt.savefig(ruta_fig + 'resid_predic_modelo8_test.pdf', dpi = 300)
plt.show()

#graficamos los residuales contra el tiempo normalizado y agregamos
#tambien las lineas de los intervalos de confianza 95% alrededor de la media cero
resid_std = (resid_train[0] - np.mean(resid_train[0]))/np.std(resid_train[0])
plt.plot(X_train_save['Time'],resid_std,'ko', markerfacecolor = 'None')
plt.axhline(1.96, color = 'b', linestyle='--')
plt.axhline(0, color = 'r', linestyle='--')
plt.axhline(-1.96, color = 'b', linestyle='--')
plt.legend(['Model 1'])
plt.xlabel('Time Normalized')
plt.ylabel('Standardized Residual')
plt.savefig(ruta_fig + 'resid_time_modelo1_train.pdf', dpi = 300)
plt.show()

resid_std = (resid_train[7] - np.mean(resid_train[7]))/np.std(resid_train[7])
plt.plot(X_train_save['Time'],resid_std,'ko', markerfacecolor = 'None')
plt.axhline(1.96, color = 'b', linestyle='--')
plt.axhline(0, color = 'r', linestyle='--')
plt.axhline(-1.96, color = 'b', linestyle='--')
plt.legend(['Model 8'])
plt.xlabel('Time Normalized')
plt.ylabel('Standardized Residual')
plt.savefig(ruta_fig + 'resid_time_modelo8_train.pdf', dpi = 300)
plt.show()

resid_std = (resid_test[0] - np.mean(resid_test[0]))/np.std(resid_test[0])
plt.plot(X_test_save['Time'],resid_std,'ko', markerfacecolor = 'None')
plt.axhline(1.96, color = 'b', linestyle='--')
plt.axhline(0, color = 'r', linestyle='--')
plt.axhline(-1.96, color = 'b', linestyle='--')
plt.legend(['Model 1'])
plt.xlabel('Time Normalized')
plt.ylabel('Standardized Residual')
plt.savefig(ruta_fig + 'resid_time_modelo1_test.pdf', dpi = 300)
plt.show()

resid_std = (resid_test[7] - np.mean(resid_test[7]))/np.std(resid_test[7])
plt.plot(X_test_save['Time'],resid_std,'ko', markerfacecolor = 'None')
plt.axhline(1.96, color = 'b', linestyle='--')
plt.axhline(0, color = 'r', linestyle='--')
plt.axhline(-1.96, color = 'b', linestyle='--')
plt.legend(['Model 8'])
plt.xlabel('Time Normalized')
plt.ylabel('Standardized Residual')
plt.savefig(ruta_fig + 'resid_time_modelo8_test.pdf', dpi = 300)
plt.show()


#graficamos la perdida del entrenamiento y prueba
# fig = plt.figure() 
# plt.plot(red.history.history['loss'], 'k-')
# plt.plot(red.history.history['val_loss'],'k--')
# plt.title('Función de pérdida')
# plt.ylabel('Pérdida (MSE)')
# plt.xlabel('Epocas')
# plt.legend(['Entrenamiento', 'Validación'], loc='upper right')
