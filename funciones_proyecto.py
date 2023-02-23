import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.stats.mstats import winsorize
import scipy.stats as stats
import patsy
import re
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist, pdist

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif

## Función para categorizar las variables con 10 o menos valores únicos
def categorizar(data):
    for col in data.columns:
        if data.nunique()[col] < 11:
            data[col] = data[col].astype('category')
    return data

## Función para cambiar una cadena de simbolos por "_" a los nombres de variables
def renombrar(data, simb):
    for col in data.columns:
        col2 = re.sub(simb, "_", col)
        data.rename(columns={col : col2},
                       inplace = True)
    return data

## Función para eliminar columnas
def eliminar_columnas(data, col = []):
    data = data.drop(col, axis=1)
    return data

## Función general para graficar las columnas que se deseen
def biplot(data, cols = [], obj = None):
    if cols == []:
        cols = data.columns
    for col in cols:
        if data[col].dtype.name != 'category':
            print('Variable continua : '+col)
            plt.figure(figsize=(10,3))
            plt.subplot(1,2,1)
            sns.histplot(data=data, x=col, hue=obj, bins=20)
            plt.subplot(1,2,2)
            sns.boxplot(data=data, x=col, hue=obj)
            plt.show()
        else:
            print('Variable categórica : '+col)
            sns.countplot(data=data, x=col, hue=obj)
            plt.show()
    
## Función manual de winsor con clip+quantile 
def winsorize_with_pandas(s, limits):
    """
    s : pd.Series
        Series to winsorize
    limits : tuple of float
        Tuple of the percentages to cut on each side of the array, 
        with respect to the number of unmasked data, as floats between 0. and 1
    """
    return s.clip(lower=s.quantile(limits[0], interpolation='lower'), 
                  upper=s.quantile(1-limits[1], interpolation='higher'))

## Función para gestionar outliers
def gestiona_outliers(col,clas = 'check'):
    print(col.name)
    # Condición de asimetría y aplicación de criterio 1 según el caso
    if abs(col.skew()) < 1:
        criterio1 = abs((col-col.mean())/col.std())>3
    else:
        criterio1 = abs((col-col.median())/col.mad())>8

    # Calcular primer cuartil     
    q1 = col.quantile(0.25)  
    # Calcular tercer cuartil  
    q3 = col.quantile(0.75)
    # Calculo de IQR
    IQR=q3-q1
    # Calcular criterio 2 (general para cualquier asimetría)
    criterio2 = (col<(q1 - 3*IQR))|(col>(q3 + 3*IQR))
    lower = col[criterio1&criterio2&(col<q1)].count()/col.dropna().count()
    upper = col[criterio1&criterio2&(col>q3)].count()/col.dropna().count()
    # Salida según el tipo deseado
    if clas == 'check':
        return(lower*100,upper*100,(lower+upper)*100)
    elif clas == 'winsor':
        return(winsorize_with_pandas(col,(lower,upper)))
    elif clas == 'miss':
        print('\n MissingAntes: ' + str(col.isna().sum()))
        col.loc[criterio1&criterio2] = np.nan
        print('MissingDespues: ' + str(col.isna().sum()) +'\n')
        return(col)
def imputar_NA(data, cols, imputer):
    #Rellena los NAs a través de los vecinos 'cols'
    #Nos servirá para todas las variables que estén tengan NAs
    X = data[cols]
    X_imputed = imputer.fit_transform(X)
    data2 = data.copy()
    data2[cols] = X_imputed
    for col in cols:
        if data[col].dtype.name != 'category':
            print('Variable continua : '+col)
            plt.figure(figsize=(10,3))
            plt.subplot(1,2,1)
            sns.histplot(data=data, x=col, bins=20)
            print('Variable continua escalada : '+col)
            plt.subplot(1,2,2)
            sns.histplot(data=data2, x=col, bins=20)
            plt.show()
        else:
            print('Variable categórica : '+col)
            plt.figure(figsize=(10,3))
            plt.subplot(1,2,1)
            sns.countplot(data=data, x=col)
            plt.subplot(1,2,2)
            print('Variable categórica escalada: '+col)
            sns.countplot(data=data2, x=col)
            plt.show()
    return data2

# Función para calcular VCramer 
def cramers_v(var1, varObj):
    if not var1.dtypes.name == 'category':
        var1 = pd.cut(var1, bins = 5)
    if not varObj.dtypes.name == 'category': 
        varObj = pd.cut(varObj, bins = 5)        
    data = pd.crosstab(var1, varObj).values
    vCramer = stats.contingency.association(data, method = 'cramer')
    return vCramer
## Función mejor tranformación
# Busca la transformación de variables input de intervalo que maximiza la VCramer o 
# la correlación tipo Pearson con la objetivo
def standard_scaler (f, c):
    sclr = StandardScaler()
    scld = sclr.fit_transform(f[[c]])
    return scld

def mejorTransf (ft, col, target, name=False, tipo = 'cramer', graf=False):

    # Escalado de datos (evitar fallos de tamaño de float64 al hacer exp de número grande)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(ft[[col]])
    vv = []
    for i in scaled:
        vv.append(float(i))
    vv = pd.Series(vv, name=col)
    # Traslación a valores positivos de la variable (sino fallaría log y las raíces)
    vv = vv + abs(min(vv))+0.0001

    # Definimos y calculamos las transformaciones típicas  
    transf = pd.DataFrame({vv.name + '_ident': vv,
                           vv.name + '_log': np.log(vv), 
                           vv.name + '_exp': np.exp(vv), 
                           vv.name + '_sqrt': np.sqrt(vv), 
                           vv.name + '_sqr': np.square(vv),
                           vv.name + '_cuarta': vv**4, 
                           vv.name + '_raiz4': vv**(1/4)})

    # Distinguimos caso cramer o caso correlación
    if tipo == 'cramer':
      # Aplicar la función cramers_v a cada transformación frente a la respuesta
        tablaCramer = pd.DataFrame(transf.apply(lambda x: cramers_v(x, target)), 
                                   columns=['VCramer'])

      # Si queremos gráfico, muestra comparativa entre las posibilidades
    if graf: px.bar(tablaCramer,
                    x=tablaCramer.VCramer,
                    title='Relaciones frente a ' + target.name).update_yaxes(categoryorder="total ascending").show()
      # Identificar mejor transformación
    best = tablaCramer.query('VCramer == VCramer.max()').index
    ser = transf[best[0]].squeeze()

    if tipo == 'cor':
        
      # Aplicar coeficiente de correlación a cada transformación frente a la respuesta
        tablaCorr = pd.DataFrame(transf.apply(lambda x: np.corrcoef(x,target)[0,1]),
                                columns=['Corr'])
      # Si queremos gráfico, muestra comparativa entre las posibilidades
        if graf : px.bar(tablaCorr,
                         x=tablaCorr.Corr,
                         title='Relaciones frente a ' + target.name).update_yaxes(categoryorder="total ascending").show()
      # identificar mejor transformación
        best = tablaCorr.query('Corr.abs() == Corr.abs().max()').index
        ser = transf[best[0]].squeeze()

    # Aquí distingue si se devuelve la variable transformada o solamente el nombre de la transformación
    if name:
        return(ser.name)
    else:
        return(ser)

def escalar(ft, col, scaler=StandardScaler()):
    scaled = scaler.fit_transform(ft[col])
    return scaled
# Función para generar la fórmula
def ols_formula(df, dependent_var, *excluded_cols):
    df_columns = list(df.columns.values)
    df_columns.remove(dependent_var)
    for col in excluded_cols:
        df_columns.remove(col)
    return dependent_var + ' ~ ' + ' + '.join(df_columns)
def tr_tst_eval_log(formula, data):
    # Generamos las matrices de diseño según la fórmula de modelo completo
    y, X = patsy.dmatrices(formula, data, return_type='dataframe')

    # Creamos 4 objetos: predictores para tr y tst y variable objetivo para tr y tst. 
    X_tr, X_tst, y_tr, y_tst = train_test_split(X, y, test_size=0.2, random_state=1234)

    # Definición de modelo
    modelo = LogisticRegression(max_iter=1000)

    # Arreglar y para que le guste a sklearn...numeric
    y_tr_ = y_tr.iloc[:,0].ravel()

    # Ajuste de modelo
    modelLog = modelo.fit(X_tr,y_tr_)

    # Accuracy del modelo en training
    acc = modelLog.score(X_tr,y_tr_)
    print('Accuracy en training: ',acc, '\n')

    # Predicciones en test
    y_pred = modelLog.predict(X_tst)

    # Matriz de confusion de clasificación 
    print('Matriz de confusión y métricas derivadas: \n',metrics.confusion_matrix(y_tst,y_pred))

    # Reporte de clasificación 
    print(metrics.classification_report(y_tst,y_pred))

    # Extraemos el Area bajo la curva ROC
    print('Area bajo la curva ROC training: \n', metrics.roc_auc_score(y_tr, modelLog.predict_proba(X_tr)[:, 1]))
    print('Area bajo la curva ROC test: \n', metrics.roc_auc_score(y_tst, modelLog.predict_proba(X_tst)[:, 1]))
#Función para comparación por validación cruzada
def cross_val_model(X, y, name, repeated=False,
                    model_df = pd.DataFrame({
                        'Model': pd.Series(dtype='str'),
                        'Accuracy': pd.Series(dtype='float')
                        }),  
                      model=LogisticRegression(random_state=seed), 
                      seed=seed):
    if repeated == True:
      cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=seed)
    else:
      cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    # Obtenemos los resultados de R2 para cada partición tr-tst
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv)
    for score in scores:
      model_df.loc[len(model_df)] = [name, score]
    # Sesgo y varianza
    plt.figure(figsize=(8,8))
    sns.boxplot(
            x=model_df.columns[0],
            y=model_df.columns[1],
            data=model_df, 
            palette='viridis')
    plt.show()
    return model_df
# Función pto de corte por Youden
def cutoff_youden(test,pred):
    fpr, tpr, thresholds = metrics.roc_curve(test,pred)
    j_scores = tpr-fpr
    j_ordered = sorted(zip(j_scores,thresholds))
    return j_ordered[-1][1]
def corte_funcion(y, cort):
    for i in y:
        if y[i] > cort:
            y[i] = 1
        else:
            y[i] = 0
    return y

# feature selection
def select_features(X_tr, y_tr, X_tst):
    fs = SelectKBest(score_func=mutual_info_classif, k='all')
    fs.fit(X_tr, y_tr)
    X_tr_fs = fs.transform(X_tr)
    X_tst_fs = fs.transform(X_tst)
    return X_tr_fs, X_tst_fs, fs