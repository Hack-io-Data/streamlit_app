


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
#from streamlit_dynamic_filters import DynamicFilters
# %%
st.title('Comprobacion de metricas')

# creamoms un boton para cargar el Los datos
upload_file = st.file_uploader('Sube Archivo', type=['csv','xlsx'])

if upload_file is not None:
    df = pd.read_csv(upload_file)
    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0', axis=1, inplace=True)
    
    st.write("archivo subido")
    st.display_file(upload_file)
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
            s = setup(df_train, target='customer_lifetime_value')

            # Crear un modelo Random Forest para regresión
            model_rf = create_model('rf')

            # Hacer predicciones en el conjunto de datos
            predictions = predict_model(model_rf, data=df_test)

            result = pull()
            print(result['RMSE'][0])

            # Metricas
            metrica = result['RMSE']
            print(f'el valor de la metrica es: {metrica}')
            st.markdown(f'# El valor de la metrica es: {metrica[0]}')
            # %%