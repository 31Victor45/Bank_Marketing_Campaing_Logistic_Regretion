import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import os

# --- Carga de Datos y Preprocesamiento del Modelo  ---
# Cargar el DataFrame
df_bank_log_model = pd.read_csv("p_items/bank_ready_for_model.csv")

# Convertir 'day' a categoría (como se hizo en el notebook y se indicó en la descripción)
df_bank_log_model["day"] = df_bank_log_model["day"].astype("category")

# Definición de Variables Predictoras (X) y Variable Objetivo (y)
numerical_features = ['balance_log', 'campaign_log', 'pdays_dias_transcurridos_log', 'previous_log']
categorical_features = ['job', 'marital', 'education', 'housing', 'loan', 'contact', 'day', 'month', 'poutcome', 'hubo_contacto_?']
target_variable = 'get_account_?'

X = df_bank_log_model[numerical_features + categorical_features]
y = df_bank_log_model[target_variable]
y = y.map({'no': 0, 'yes': 1})

# Separación de Datos (solo para entrenar el pipeline y el modelo)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Preprocesamiento de Datos (Pipelines)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Creación y Entrenamiento del Modelo de Regresión Logística
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced'))])

model_pipeline.fit(X_train, y_train)

# --- Funciones para la aplicación ---

def get_unique_values(df, col_name, sort=False, custom_sort_order=None):
    """Obtiene los valores únicos de una columna categórica y opcionalmente los ordena.
    Permite un orden de clasificación personalizado."""
    values = df[col_name].unique().tolist()
    if sort:
        if custom_sort_order:
            # Ordenar según un orden personalizado predefinido
            # Esto maneja casos donde los valores pueden ser cadenas (como abreviaturas de meses)
            order_map = {item: i for i, item in enumerate(custom_sort_order)}
            # Usa order_map.get(x, len(custom_sort_order)) para colocar elementos no mapeados al final
            return sorted(values, key=lambda x: order_map.get(x, len(custom_sort_order)))
        else:
            try:
                # Intenta ordenar numéricamente si es posible (útil para 'day')
                # Asegura que sean enteros antes de ordenar
                return sorted([int(x) for x in values if str(x).isdigit()])
            except ValueError:
                # Si no es numérico, ordena alfabéticamente
                return sorted(values)
    return values

def apply_log_transform_for_balance(value):
    """
    Aplica la transformación logarítmica a 'balance'.
    Crucial: Esta función debe replicar exactamente cómo se manejaron los valores negativos
    para la columna 'balance_log' en tu dataset 'bank_ready_for_model.csv'.
    """
    if value >= 0:
        return np.log1p(value)
    else:
        # Si balance_log en tu dataset original fue np.log1p(abs(balance_original)) para negativos:
        return np.log1p(abs(value))
        # Si balance_log en tu dataset original fue np.log1p(0) para cualquier balance <= 0:
        # return np.log1p(0)


def apply_log_transform_general(value):
    """
    Aplica la transformación logarítmica np.log1p (log(1+x)) a un valor numérico.
    Esta función es robusta para valores cero, donde log1p(0) = 0.
    """
    return np.log1p(value)

# --- Streamlit UI ---

st.set_page_config(page_title="Clasificador de Apertura de Cuenta Bancaria", layout="centered")

st.title("Clasificador de Apertura de Cuenta Bancaria de Ahorro 🏦")
st.markdown("Ingresa los datos del cliente para predecir si abrirá una cuenta.")

st.sidebar.header("Parámetros de Entrada del Cliente")

# --- Inputs para variables numéricas ---
st.sidebar.subheader("Variables Numéricas")

# balance: Rango de -4000 a 80000, con valor inicial 1000
balance = st.sidebar.number_input("Balance promedio anual (euros)", min_value=-4000.0, max_value=80000.0, value=1000.0, step=100.0)
campaign = st.sidebar.number_input("Número de contactos realizados durante la campaña (incl. el actual)", min_value=1, max_value=60, value=2, step=1)
pdays = st.sidebar.number_input("Días transcurridos desde el último contacto de una campaña anterior (-1 si nunca)", min_value=-1, max_value=999, value=100, step=1)
previous = st.sidebar.number_input("Número de contactos realizados antes de esta campaña", min_value=0, max_value=30, value=0, step=1)


# --- Inputs para variables categóricas ---
st.sidebar.subheader("Variables Categóricas")

job_options = get_unique_values(df_bank_log_model, 'job')
job = st.sidebar.selectbox("Ocupación", job_options)

marital_options = get_unique_values(df_bank_log_model, 'marital')
marital = st.sidebar.selectbox("Estado Civil", marital_options)

education_options = get_unique_values(df_bank_log_model, 'education')
education = st.sidebar.selectbox("Nivel Educativo", education_options)

housing_options = get_unique_values(df_bank_log_model, 'housing')
housing = st.sidebar.selectbox("¿Tiene préstamo hipotecario?", housing_options)

loan_options = get_unique_values(df_bank_log_model, 'loan')
loan = st.sidebar.selectbox("¿Tiene préstamo personal?", loan_options)

contact_options = get_unique_values(df_bank_log_model, 'contact')
contact = st.sidebar.selectbox("Tipo de contacto de la última campaña", contact_options)

# 'day' se obtiene y se ordena numéricamente
day_options = get_unique_values(df_bank_log_model, 'day', sort=True)
day = st.sidebar.selectbox("Día del mes del último contacto", day_options)

# Definir el orden cronológico de los meses, empezando por Mayo
month_chronological_order = ['may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 'jan', 'feb', 'mar', 'apr']
month_options = get_unique_values(df_bank_log_model, 'month', sort=True, custom_sort_order=month_chronological_order)
month = st.sidebar.selectbox("Mes del último contacto", month_options)

poutcome_options = get_unique_values(df_bank_log_model, 'poutcome')
poutcome = st.sidebar.selectbox("Resultado de la campaña anterior", poutcome_options)

hubo_contacto_options = get_unique_values(df_bank_log_model, 'hubo_contacto_?')
hubo_contacto = st.sidebar.selectbox("¿Fue contactado en una campaña anterior?", hubo_contacto_options)

# --- Preparar datos para la predicción ---
# Aplicar transformaciones logarítmicas a las entradas numéricas
balance_log_val = apply_log_transform_for_balance(balance) # Usar la función específica para balance
campaign_log_val = apply_log_transform_general(campaign)

# Manejo específico para pdays: si es -1, se transforma a 0 antes de log1p
pdays_for_log = pdays if pdays != -1 else 0
pdays_dias_transcurridos_log_val = apply_log_transform_general(pdays_for_log)

previous_log_val = apply_log_transform_general(previous)

# Crear un DataFrame con las entradas del usuario
input_data = pd.DataFrame([[
    balance_log_val, campaign_log_val, pdays_dias_transcurridos_log_val, previous_log_val,
    job, marital, education, housing, loan, contact, day, month, poutcome, hubo_contacto
]], columns=numerical_features + categorical_features)

# Asegurarse de que 'day' sea de tipo 'category' en el DataFrame de entrada para OneHotEncoder
# esto es importante porque el OneHotEncoder fue entrenado con 'day' como categoría
input_data['day'] = input_data['day'].astype('category')


# --- Realizar Predicción ---
if st.sidebar.button("Clasificar Cliente"):
    st.subheader("Resultado de la Clasificación:")

    # Realizar la predicción
    prediction = model_pipeline.predict(input_data)[0]
    prediction_proba = model_pipeline.predict_proba(input_data)[:, 1][0] # Probabilidad de la clase positiva (1, 'yes')

    # Centrar y reducir el tamaño de la imagen
    col1, col2, col3 = st.columns([1,2,1]) # Crea 3 columnas, la del medio es más ancha para la imagen

    if prediction == 1: # Predice 'yes' (abrirá cuenta)
        st.success(f"**¡El cliente califica para abrir una cuenta!**")
        st.write(f"Probabilidad de abrir cuenta: **{prediction_proba:.2f}**")
        # Mostrar imagen de califica
        with col2: # Coloca la imagen en la columna central
            if os.path.exists("p_items/califica.png"):
                st.image("p_items/califica.png", caption="¡Cliente que califica!", use_container_width=True)
            else:
                st.warning("Imagen 'califica.png' no encontrada. Asegúrate de que esté en la misma carpeta que 'app.py'.")
    else: # Predice 'no' (no abrirá cuenta)
        st.error(f"**El cliente NO califica para abrir una cuenta.**")
        st.write(f"Probabilidad de abrir cuenta: **{prediction_proba:.2f}**")
        # Mostrar imagen de no_califica
        with col2: # Coloca la imagen en la columna central
            if os.path.exists("p_items/no_califica.png"):
                st.image("p_items/no_califica.png", caption="Cliente que no califica.", use_container_width=True)
            else:
                st.warning("Imagen 'no_califica.png' no encontrada. Asegúrate de que esté en la misma carpeta que 'app.py'.")

    st.markdown("---")
    st.write("### Consideraciones del Modelo:")
    st.markdown("""
    Este modelo de Regresión Logística fue optimizado para la **precisión en la identificación de clientes que NO abrirán una cuenta** (precisión del 93% para la clase 'no').
    Esto es crucial para **optimizar los recursos** de la empresa al evitar contactos ineficientes.

    Sin embargo, la **precisión para la clase 'yes' es más baja (22%)** con el umbral por defecto. Esto significa que, si el modelo predice un 'sí', existe una probabilidad significativa de que sea un falso positivo. La decisión de mantener este balance se basa en la estrategia de negocio de **priorizar la minimización del desperdicio de recursos** sobre la maximización de la captura de todos los posibles 'yes', dado el costo asociado a los falsos positivos.

    **Nota Importante:** Si la probabilidad de que el cliente abra una cuenta es del **90% o superior**, esta predicción individual se considera **altamente fiable**. Esto sugiere que el cliente comparte características muy sólidas con los casos de éxito claros que el modelo ha identificado, permitiendo una asignación de recursos aún más precisa.
    """)