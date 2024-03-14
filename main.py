


# %%
# Tratamiento de datos y tablas de datos
# ------------------------------------------------------------------------------
import numpy as np
import pandas as pd

# Modelado y evaluación
# ------------------------------------------------------------------------------
from pycaret.regression import *
from pycaret.regression import evaluate_model

# Aplicacion 
# ------------------------------------------------------------------------------
import streamlit as st
from streamlit_dynamic_filters import DynamicFilters
# %%
st.title('Comprobacion de metricas')

# creamoms un boton para cargar el Los datos
upload_file = st.file_uploader('Sube Archivo', type=['csv'])

if upload_file is not None:
    df = pd.read_csv(upload_file)
    st.write("archivo subido")
    #list(df.columns)


    #filtered = st.multiselect("Selecciona las columnas que no quieras en tu modelo", options=list(df.columns), default=list(df.columns))

    #st.write(df[filtered])
    objetivo = st.selectbox('Selecciona la variable dependiente ',
                            (list(df.columns)),
                            index=None,
                            placeholder="Select contact method...")
    
    if objetivo is not None:
    
        numericas  = list(df.select_dtypes(include = np.number ).columns.drop(objetivo))

        categoricas = list(df.select_dtypes(exclude = 'number').columns)
        
        ignorar = []
            
        st.markdown('# Datos y variables cargadas con exito')

        

        if st.button("Iniciar analisis", type="primary"):
            

            # definimos conjuntos de entrenamiento y test
            df_train = df[:-20]
            df_test = df[-20:]

            # generamos objeto de experimento
            exp1 = RegressionExperiment()
            exp1.setup(df_train, ignore_features=ignorar, target=objetivo, 
                        session_id=42, train_size=0.7, 
                        categorical_features=categoricas, numeric_features=numericas)


            # seleccionamos el mejor modelo 
            top1_exp1 = exp1.compare_models(n_select=1)

            # entrenamos el modelo 
            modelo_final = exp1.create_model(top1_exp1)


            # guardamos el modelo 
            exp1.save_model(modelo_final, 'output/my_pycaret_regression')

            # Cargamos el modelo
            my_winning_regressor = load_model('output/my_pycaret_regression')
            s = setup(data = df_test,  target = 'customer_lifetime_value')

            # Crear un modelo 
            modelo = create_model(my_winning_regressor)

            # Predicciones
            pred = predict_model(my_winning_regressor, data=df_train)
            result = pull()

            # Metricas
            metrica = result['RMSE']
            print(f'el valor de la metrica es: {metrica}')
            st.markdown(f'# El valor de la metrica es: {metrica[0]}')
            # %%