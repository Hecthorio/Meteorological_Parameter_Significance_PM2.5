# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 19:47:23 2023

@author: hecto
"""

#script para generar un analsis exploratorio de los datos

#importamos las librerias
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
import statsmodels.api as sm

#damos de alta la ruta y el nombre del df completo
ruta = 'C:/Users/hecto/OneDrive/Documentos/ITA/Posdoc Proyecto/articulo 1/codigos/base_datos/datos_unidos/datos_final.csv'

#damos de alta la ruta donde se guardaran las figuras generadas
ruta_fig = 'C:/Users/hecto/OneDrive/Documentos/ITA/Posdoc Proyecto/articulo 1/'

#leemos el archivo y lo pasamos a un df
df = pd.read_csv(ruta)

#cambiamos los nombres de los parametros al íngles
df.rename(columns = {'Hora':'Time', 'DV':'WD', 'HR':'RH', 'PM25':'PM2.5(t)', 
                     'VV':'WS', 'PM25_t':'PM2.5(t-1)'}, inplace = True)

#imprimimos la tabla con la información de los parámetros del modelo
info = df.describe()

#transponesmos la tabla para que quepa
info = info.T

#redondeamos todos los valores
info = info.round(4)

#eliminamos la columna de conteo
info.drop(columns='count', inplace=True)

#agregamos una columna con las unidades de cada parametro
info = info.assign(Units = ['-', 'Hours', 'Degrees', '%', 'ug/m3', 'mm', '°C', 'm/s', '-', 'ug/m3'])

#movemos a la 1ra columna las unidades
info = info[['Units','mean', 'std', 'min', '25%', '50%', '75%', 'max']]

#generamos la tabla de latex
tabla = tabulate(info, tablefmt='latex', headers='keys', showindex=True)

###############################################################################
###############################################################################

#graficamos el pairgrid
def corrdot(*args, **kwargs):
    corr_r = args[0].corr(args[1], 'pearson')
    corr_text = f"{corr_r:2.2f}".replace("0.", ".")
    ax = plt.gca()
    ax.set_axis_off()
    marker_size = abs(corr_r) * 100000
    ax.scatter([.5], [.5], marker_size, [corr_r], alpha=0.6, cmap="coolwarm",
               vmin=-1, vmax=1, transform=ax.transAxes)
    font_size = abs(corr_r) * 100 + 5
    ax.annotate(corr_text, [.5, .5,],  xycoords="axes fraction",
                ha='center', va='center', fontsize=font_size)

sns.set(style='white', font_scale=2)
g = sns.PairGrid(df, aspect=1, diag_sharey=False) #1.4
#g.map_lower(sns.regplot, lowess=True, ci=False, line_kws={'color': 'red'}, scatter_kws={'edgecolor': 'k', 'facecolor':'white', 's': 20})
g.map_lower(sns.regplot, lowess=True, ci=False, line_kws={'color': 'red'}, scatter_kws={'edgecolor': 'k', 'facecolor':'None', 's': 100})
g.map_diag(sns.distplot, kde_kws={'color': 'red'}, hist_kws={'histtype': 'bar', 'lw': 2,'edgecolor': 'k', 'facecolor':'grey'})
g.map_upper(corrdot)

#guardamos la figura
plt.savefig(ruta_fig + 'pailpairs.jpg', dpi=300)


###############################################################################
###############################################################################
#generamos las funciones de ACF y PACF, para esto separamos el valor
#de los PM2.5 en otra variable
residuales = df['PM2.5(t)']

# Crear la figura y los subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Autocorrelación de los residuales
sm.graphics.tsa.plot_acf(residuales, lags=24, ax=ax1)
#ax1.set_title('Autocorrelación error (entrenamiento)')
ax1.set_xlabel('Lag')
ax1.set_ylabel('ACF')
#ax1.set_ylim(-0.5, 0.5)

# Autocorrelación parcial de los residuales
sm.graphics.tsa.plot_pacf(residuales, lags=24, ax=ax2)
#ax2.set_title('Autocorrelación parcial error (entrenamiento)')
ax2.set_xlabel('Lag')
ax2.set_ylabel('PACF')
#ax2.set_ylim(-0.5, 0.5)

#guardamos la figura
plt.savefig(ruta_fig + 'acf_and_pacf.pdf', dpi=300)

# Mostrar la figura con ambos subplots
plt.show()

#generamos el subplot de las particulas para analizar los outliners
plt.figure(figsize = (15,6), dpi = 300)
sns.boxplot(x=df['Time'], y=df['PM2.5(t)'])
plt.savefig(ruta_fig + 'boxplot_PM.pdf', dpi = 300)

