import pandas as pd
from sklearn import linear_model
import statsmodels.api as sm
from datetime import datetime, timedelta, date
import os
os.chdir(r"C:\Users\andre\Desktop\scripts\DISTRITOS\Coimbra")

# Leitura do CSV com parse das datas
df = pd.read_csv (r'coimbra.csv', sep=';', thousands=',', parse_dates=['DATA'], dayfirst=True) #, header=None, names=headers, dtype=dtypes
#print (df)

# Seleção da data base para a predição
#start_date = input("Qual a data para estimativa de casos (aaaa-mm-dd):")
#start_date = datetime.strptime(start_date, "%Y-%m-%d")

# opening a file in 'w' para escrever os resultados
file = open('output_step14.txt', 'w')
file.writelines("Data;Estimado;R^2\n")
file2 = open('predicted_step14.txt', 'w')
file2.writelines("Data;Predicted\n")
#Criar lista de modelos de 14 em 14 dias
start = 0
stop = len(df)
step = 14
modelos14dias = list(range(start, stop, step))
#print(list(range(start, stop, step)))

# Ciclo para estimar todas os dias existentes na BD com 14 dias de histórico 
for i, row in df.iterrows():

    #print("i: " + str(i))
    
    start_date = df.at[i,'DATA']
    #print("A data para estimativa a 14 dias é: " + str(start_date))
    
    date_1 = pd.to_datetime(start_date) + pd.DateOffset(days=-29)
    #print("A data 1 é: " + str(date_1))
    end_date1 = pd.to_datetime(date_1) + pd.DateOffset(days=+15)
    #print("O fim do período 1 é: " + str(end_date1))
    df_X = df[(df['DATA'] > date_1) & (df['DATA'] < end_date1)]
    if not df_X.empty:
        #print (df_X)
        date_2 = pd.to_datetime(start_date) + pd.DateOffset(days=-15)
        end_date2 = pd.to_datetime(start_date)
        #print("A referência temporal de NC é: " + str(date_2) + str(end_date2))
        df_Y = df[(df['DATA'] > date_2) & (df['DATA'] < end_date2)]
        #print (df_Y)
        date_3 = pd.to_datetime(start_date)
        end_date3 = pd.to_datetime(start_date) + pd.DateOffset(days=+15)
        df_Y2 = df[(df['DATA'] > date_3) & (df['DATA'] < end_date3)]
        #print (df_Y2)
        if not df_Y.empty and len(df_X)==len(df_Y) and i in modelos14dias:
            print(f"Index: {i}")
            X = df_X[['RR','GPH','P','T','WP','RS']] # workplaces_percent_change_from_baseline	residential_percent_change_from_baseline, retail_and_recreation_percent_change_from_baseline, grocery_and_pharmacy_percent_change_from_baseline, parks_percent_change_from_baseline,	transit_stations_percent_change_from_baseline            
            Y = df_Y['NC'] # Número de casos diários
            print (df_X)
            print (df_Y)
            # with sklearn
            regr = linear_model.LinearRegression(fit_intercept=True)
            regr.fit(X, Y)
            
            #print('Intercept: \n', regr.intercept_)
            #print('Coefficients: \n', regr.coef_)
            # prediction with sklearn
            predicted_date = start_date
            #print("A estimativa refere-se ao dia: " + str(predicted_date))
            df_predicted = df[df['DATA'] == predicted_date]
            #print (df_predicted)
            #print (df_Y2)
            # Valores das variáveis para os 14 dias a estimar
            estimado = 0
            observado = 0
            erro_obs = 0
            #Ciclo para estimar o número de casos para o dia n de um conjunto de 14
            for i, row in df_X.iterrows():
                print('Intercept: \n', regr.intercept_)
                print('Coefficients: \n', regr.coef_)
                print(f'r_sqr value: {regr.score(X, Y)}')
                #print(f"Index: {i}")
                #print(f"{row}\n")
                index_predicted = {i}
                predicted_DATA = df_X.at[i,'DATA'] #data em X
                predicted_DATAMAIS14 = pd.to_datetime(predicted_DATA) + pd.DateOffset(days=+14)
                default_RR = df_X.at[i,'RR'] #new retail_and_recreation_percent_change_from_baseline
                default_GPH = df_X.at[i,'GPH'] #new rocery_and_pharmacy_percent_change_from_baseline
                default_P = df_X.at[i,'P'] #New parks_percent_change_from_baseline
                default_T = df_X.at[i,'T'] #New transit_stations_percent_change_from_baseline
                default_WP = df_X.at[i,'WP'] #New workplaces_percent_change_from_baseline
                default_RS = df_X.at[i,'RS'] #new residential_percent_change_from_baseline                           
                print("DATA: " + str(predicted_DATAMAIS14))
                estimado = int(regr.predict([[default_RR , default_GPH, default_P, default_T, default_WP, default_RS]]))
                print("ESTIMADO1: " + str(estimado))
                file.writelines(str (predicted_DATAMAIS14) + ';' + str (estimado) + ';' + str (regr.score(X, Y)) + '\n')
            for i, row in df_Y.iterrows():
                default_RR2 = df_Y.at[i,'RR'] #new retail_and_recreation_percent_change_from_baseline
                default_GPH2 = df_Y.at[i,'GPH'] #new rocery_and_pharmacy_percent_change_from_baseline
                default_P2 = df_Y.at[i,'P'] #New parks_percent_change_from_baseline
                default_T2 = df_Y.at[i,'T'] #New transit_stations_percent_change_from_baseline
                default_WP2 = df_Y.at[i,'WP'] #New workplaces_percent_change_from_baseline
                default_RS2 = df_Y.at[i,'RS'] #new residential_percent_change_from_baseline 
                predicted_DATA2 = df_Y.at[i,'DATA'] #data em Y
                predicted_DATA2MAIS14 = pd.to_datetime(predicted_DATA2) + pd.DateOffset(days=+14)
                print("DATA2: " + str(predicted_DATA2MAIS14))
                estimado2 = int(regr.predict([[default_RR2 , default_GPH2, default_P2, default_T2, default_WP2, default_RS2]]))
                print("ESTIMADO2: " + str(estimado2))             
                file2.writelines(str (predicted_DATA2MAIS14) + ';' + str (estimado2) + '\n')
                #print ('Estimado row: ' +str(df['DATA']) +'|'+str(estimado))
            #Ciclo para estimar o número de casos para o dia n de um conjunto de 14
            for i, row in df_Y2.iterrows():            
                observado = observado + int(df_Y2.at[i,'NC'])                
            residuo = observado - estimado
            print ('Observado: ' + str(observado))
            print ('Estimado: ' + str(estimado))
            # writing data using the write() method
            #file.writelines(str (observado) + ';' + str (estimado) + ';' + str (residuo) + ';' + str({regr.score(X, Y)}) + '\n')
# closing the file
file.close()
file2.close()           
print ("Ficheiro com as estimativas gerado!")


