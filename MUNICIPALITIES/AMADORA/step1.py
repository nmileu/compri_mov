import pandas as pd
from sklearn import linear_model
import statsmodels.api as sm
from datetime import datetime, timedelta, date
import numpy as np
import os
os.chdir(r"C:\Users\andre\Desktop\scripts\AMADORA")

# Leitura do CSV com parse das datas
df = pd.read_csv (r'amadora.csv', sep=';', thousands=',', parse_dates=['DATA'], dayfirst=True) #, header=None, names=headers, dtype=dtypes
#print (df)

# Seleção da data base para a predição
#start_date = input("Qual a data para estimativa de casos (aaaa-mm-dd):")
#start_date = datetime.strptime(start_date, "%Y-%m-%d")

# opening a file in 'w' para escrever os resultados
file = open('output_step1.txt', 'w')
file.writelines("Data;Estimado;R^2\n")
file2 = open('predicted_step1.txt', 'w')
file2.writelines("Data;Predicted\n")
# Ciclo para estimar todas os dias existentes na BD com 14 dias de histórico 
for i, row in df.iterrows():
    start_date = df.at[i,'DATA']
    #print("A data para estimativa a 14 dias é: " + str(start_date))
    
    date_1 = pd.to_datetime(start_date) + pd.DateOffset(days=-29)
    #print("A data 1 é: " + str(date_1))
    end_date1 = pd.to_datetime(date_1) + pd.DateOffset(days=+15)
    #print("O fim do período 1 é: " + str(end_date1))
    df_X = df[(df['DATA'] > date_1) & (df['DATA'] < end_date1)]
    if not df_X.empty:
        print (df_X)
        date_2 = pd.to_datetime(start_date) + pd.DateOffset(days=-15)
        end_date2 = pd.to_datetime(start_date)
        #print("A referência temporal de NC é: " + str(date_2))
        df_Y = df[(df['DATA'] > date_2) & (df['DATA'] < end_date2)]
        print (df_Y)
        date_3 = pd.to_datetime(start_date)
        end_date3 = pd.to_datetime(start_date) + pd.DateOffset(days=+15)
        #print("A referência temporal de NC é: " + str(end_date3))
        df_Y2 = df[(df['DATA'] > date_3) & (df['DATA'] < end_date3)]
        #print (df_Y2)
        if not df_Y.empty and len(df_X)==len(df_Y):
            X = df_X[['RR','GPH','P','T','WP','RS']] # workplaces_percent_change_from_baseline	residential_percent_change_from_baseline, retail_and_recreation_percent_change_from_baseline, grocery_and_pharmacy_percent_change_from_baseline, parks_percent_change_from_baseline,	transit_stations_percent_change_from_baseline            
            Y = df_Y['NC'] # Número de casos diários
             
            # with sklearn
            regr = linear_model.LinearRegression(fit_intercept=True)
            regr.fit(X, Y)

            print('Intercept: \n', regr.intercept_)
            print('Coefficients: \n', regr.coef_)
            # prediction with sklearn
            predicted_date = start_date
            #print("A estimativa refere-se ao dia: " + str(predicted_date))
            df_predicted = df[df['DATA'] == predicted_date]
            #print (df_predicted)
            # Valores das variáveis para os 14 dias a estimar
            estimado = 0
            observado = 0
            erro_obs = 0
            
            #print (df_X[['RR', 'GPH', 'P','T','WP','RS']].iloc[[0]])
            default_RR = int (df_X['RR'].iloc[[0]]) #new retail_and_recreation_percent_change_from_baseline
            default_GPH = int (df_X['GPH'].iloc[[0]]) #new rocery_and_pharmacy_percent_change_from_baseline
            default_P = int (df_X['P'].iloc[[0]]) #New parks_percent_change_from_baseline
            default_T = int (df_X['T'].iloc[[0]]) #New transit_stations_percent_change_from_baseline
            default_WP = int (df_X['WP'].iloc[[0]]) #New workplaces_percent_change_from_baseline
            default_RS = int (df_X['RS'].iloc[[0]]) #new residential_percent_change_from_baseline
            #print ("RS:" + str(default_RS))
            estimado = int(regr.predict([[default_RR , default_GPH, default_P, default_T, default_WP, default_RS]]))
            
            print (df_Y[['RR', 'GPH', 'P','T','WP','RS']].iloc[[0]])
            default_RR2 = int (df_Y['RR'].iloc[[0]]) #new retail_and_recreation_percent_change_from_baseline
            default_GPH2 = int (df_Y['GPH'].iloc[[0]]) #new rocery_and_pharmacy_percent_change_from_baseline
            default_P2 = int (df_Y['P'].iloc[[0]]) #New parks_percent_change_from_baseline
            default_T2 = int (df_Y['T'].iloc[[0]]) #New transit_stations_percent_change_from_baseline
            default_WP2 = int (df_Y['WP'].iloc[[0]]) #New workplaces_percent_change_from_baseline
            default_RS2 = int (df_Y['RS'].iloc[[0]]) #new residential_percent_change_from_baseline
            #print ("RS:" + str(default_RS))
            estimado2 = int(regr.predict([[default_RR2 , default_GPH2, default_P2, default_T2, default_WP2, default_RS2]]))
            print ("Estimado2: " + str(estimado2)) 
            
            datareferencia = pd.to_datetime(date_2) + pd.DateOffset(days=+1)
            datareferencia2 = pd.to_datetime(date_2) + pd.DateOffset(days=+15)
            print("A referência temporal do estimado é: " + str(datareferencia))
            #estimado = int(regr.predict([[default_RR , default_GPH, default_P, default_T, default_WP, default_RS]]))            
            print ('Estimado: ' + str(estimado))
            
            #Ciclo para estimar o número de casos para o dia n de um conjunto de 14
            for i, row in df_Y2.iterrows():            
                observado = observado + int(df_Y2.at[i,'NC'])                
            residuo = observado - estimado
            
            # writing data using the write() method
            file.writelines(str (datareferencia) + ';' + str(estimado) + ';' + str (regr.score(X, Y)) + '\n')
            file2.writelines(str (datareferencia2) + ';' + str(estimado2) + '\n')
# closing the file
file.close()           
print ("Ficheiro com as estimativas gerado!")


