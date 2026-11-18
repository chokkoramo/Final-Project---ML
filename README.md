# 🚗 Tasador y Buscador de Vehículos Usados

Aplicación web desarrollada en **Streamlit** que integra modelos de **Machine Learning** (Regresión Lineal y K-Nearest Neighbors) y un **Chatbot Asistente Inteligente** impulsado por el SDK oficial de **Google GenAI (Gemini)** con fallback offline transparente basado en expresiones regulares.

---

## 📋 Tabla de Contenidos
- [Descripción General](#-descripción-general)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Pipeline ETL y Datos](#-pipeline-etl-y-datos)
- [Fundamentación Técnica de Machine Learning](#-fundamentación-técnica-de-machine-learning)
  - [1. Regresión Lineal (OLS)](#1-regresión-lineal-ols)
  - [2. K-Nearest Neighbors (KNN) con StandardScaler](#2-k-nearest-neighbors-knn-con-standardscaler)
- [Chatbot Inteligente Híbrido](#-chatbot-inteligente-híbrido)
- [Instalación y Puesta en Marcha](#-instalación-y-puesta-en-marcha)
- [Uso de la Aplicación](#-uso-de-la-aplicación)

---

## 🌟 Descripción General

El mercado de vehículos usados presenta alta dispersión de precios debido a factores como el año de fabricación, desgaste por kilometraje y potencia del motor. Este proyecto unifica múltiples fuentes internacionales de datos vehiculares (Reino Unido e India), los normaliza a unidades y moneda de Colombia (COP), y ofrece tres interfaces principales:

1. **💬 Asistente Virtual (Chatbot):** Permite consultar precios y similares en lenguaje natural, con soporte multi-turno.
2. **🔍 Tasador Manual y Buscador KNN:** Controles interactivos con métricas de mercado comparativas.
3. **📈 Análisis y Diagnóstico:** Panel de evaluación del modelo con métricas ($R^2$, MAE, RMSE) y gráficos de regresión.

---

## 📁 Estructura del Proyecto

```text
Final Project - ML/
├── .env.example              # Plantilla para variables de entorno (GEMINI_API_KEY)
├── .env                      # Archivo local de variables de entorno (ignorado en git)
├── .gitignore                # Reglas de exclusión para git (cache, .venv, datos procesados)
├── requirements.txt          # Dependencias de Python del proyecto
├── app.py                    # Aplicación principal de Streamlit
├── README.md                 # Documentación completa y técnica del proyecto
├── data/
│   ├── raw/                  # Datasets originales
│   │   ├── audi.csv
│   │   ├── Car details v3.csv
│   │   └── CAR DETAILS FROM CAR DEKHO.csv
│   └── processed/            # Dataset limpio y unificado
│       └── vehicles_unified.csv
└── src/
    ├── __init__.py           # Paquete fuente
    ├── etl.py                # Pipeline de extracción, limpieza y consolidación
    ├── models.py             # Entrenamiento y evaluación de Regresión Lineal y KNN
    └── chatbot.py            # Orquestación conversacional (Gemini + Regex)
```

---

## 🔄 Pipeline ETL y Datos

El script `src/etl.py` consolida **17,907 vehículos limpios** a partir de tres fuentes heterogéneas:

| Dataset Original | Conversión de Moneda | Conversión de Unidades | Tratamiento Específico |
| :--- | :---: | :---: | :--- |
| `audi.csv` | GBP $\rightarrow$ COP ($\times 5400$) | Millas $\rightarrow$ Km ($\times 1.60934$), Litros $\rightarrow$ CC ($\times 1000$) | Prefijo `"Audi "` en modelo; filtro `engineSize > 0`. |
| `Car details v3.csv` | INR $\rightarrow$ COP ($\times 50$) | Extracción numérica de `" CC"` | Descarte de nulos en cilindraje. |
| `CAR DETAILS FROM CAR DEKHO.csv` | INR $\rightarrow$ COP ($\times 50$) | Regex de litros en `name` ($\times 1000$) | Descarte de registros sin cilindraje determinable. |

**Reglas de Normalización y Limpieza:**
- **Combustible:** Estandarizado a `Gasolina`, `Diésel`, `Híbrido` y `Otro`.
- **Transmisión:** Estandarizada a `Manual`, `Automática` y `Semi-Automática`.
- **Deduplicación:** Eliminación de duplicados exactos en características descriptivas (1,561 registros).
- **Outliers:** Filtro estricto de precios irreales (`price_cop <= 5,000,000 COP`) y kilometraje inválido (`km_driven < 0`).
- **Valores Nulos:** **0 nulos** en todas las 8 columnas del dataset unificado.

---

## 🧠 Fundamentación Técnica de Machine Learning

### 1. Regresión Lineal (OLS)
Utilizado para la estimación cuantitativa y paramétrica del valor comercial.

- **Variables Predictoras ($X$):**
  - `year`: Año de fabricación.
  - `km_driven`: Kilometraje recorrido (km).
  - `engine_cc`: Cilindraje del motor (cc).
- **Variable Objetivo ($y$):** `price_cop` (Precio en Pesos Colombianos).
- **Partición:** 80% entrenamiento / 20% prueba (`random_state=42`).

#### Métricas de Rendimiento:
- **$R^2$ Score:** `0.7316` (Explica el **73.16%** de la varianza en el precio).
- **MAE (Error Absoluto Medio):** `$26,729,658 COP`.
- **RMSE (Raíz del Error Cuadrático Medio):** `$36,408,305 COP`.

#### Ecuación Aprendida e Interpretación:
$$\text{Precio (COP)} = -17,833,389,061 + (8,835,531 \cdot \text{year}) - (283.11 \cdot \text{km\_driven}) + (69,623.71 \cdot \text{engine\_cc})$$

- $\beta_{\text{year}} = +\$8,835,531 \text{ COP/año}$: Refleja el costo de la depreciación anual; cada año más reciente suma ~$8.8M COP.
- $\beta_{\text{km}} = -\$283.11 \text{ COP/km}$: Modela el desgaste mecánico acumulado por el rodaje.
- $\beta_{\text{cc}} = +\$69,623.71 \text{ COP/cc}$: Cuantifica la prima de mercado asociada a mayor potencia y cilindrada.

---

### 2. K-Nearest Neighbors (KNN) con StandardScaler
Utilizado para la búsqueda de los 5 vehículos más similares del mercado.

- **Estandarización (`StandardScaler`):** Las variables tienen escalas dispares (`km_driven` en decenas de miles vs `year` alrededor de 2018). El escalamiento a media 0 y varianza unitaria es **estrictamente necesario** para evitar que el kilometraje domine completamente la distancia euclidiana.
- **Métrica:** Distancia Euclidiana sobre el espacio estandarizado:
  $$d(p, q) = \sqrt{\sum_{i=1}^3 \left(\frac{p_i - \mu_i}{\sigma_i} - \frac{q_i - \mu_i}{\sigma_i}\right)^2}$$
- **Porcentaje de Similitud Normalizado:**
  $$\text{Similitud (\%)} = \frac{100}{1 + d}$$
  (Garantiza un rango acotado entre 0% y 100%, donde distancia 0 representa identidad perfecta).

---

## 🤖 Chatbot Inteligente Híbrido

El módulo `src/chatbot.py` implementa una arquitectura resiliente:

```text
[Mensaje del Usuario]
         │
         ▼
¿Existe GEMINI_API_KEY y conexión?
   ├── SÍ ──► Extracción con LLM (google-genai / gemini-3.8-flash)
   └── NO ──► Extracción Offline con Expresiones Regulares (Regex)
         │
         ▼
¿Faltan parámetros (año, km, cc)?
   ├── SÍ ──► Pregunta amablemente por el dato faltante (Mantiene estado)
   └── NO ──► Ejecuta Regresión + KNN ──► Entrega Avalúo + Similares + Recomendación
```

---

## 🚀 Instalación y Puesta en Marcha

### Prerrequisitos
- Python 3.10 o superior instalado.

### 1. Clonar el Repositorio / Navegar a la Carpeta
```powershell
cd "~\Final Project - ML"
```

### 2. Crear y Activar el Entorno Virtual
```powershell
# En Windows (PowerShell):
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Instalar Dependencias
```powershell
pip install -r requirements.txt
```

### 4. Configurar Variables de Entorno (Opcional)
Crea o edita el archivo `.env`:
```env
GEMINI_API_KEY=tu_api_key_aqui
```
*(Si no dispones de API key, la aplicación funcionará de manera automática en Modo Offline con Regex sin restricciones).*

### 5. Generar el Dataset Procesado (si no existe)
```powershell
python src/etl.py
```

### 6. Ejecutar la Aplicación Streamlit
```powershell
streamlit run app.py
```
O directamente con el ejecutable del entorno virtual:
```powershell
.\.venv\Scripts\streamlit run app.py
```

---

## 🖥️ Uso de la Aplicación

- **Barra Lateral:** Permite ingresar o cambiar la `GEMINI_API_KEY` en caliente, ver el estado del modo IA y revisar el total de vehículos cargados.
- **Pestaña 1 (Chatbot):** Interactúa conversando o usando los botones de prueba rápida. Admite preguntas parciales para evaluar la memoria conversacional.
- **Pestaña 2 (Tasador Manual):** Mueve los deslizadores de año, km y cc para ver el avalúo instantáneo, la comparación frente al promedio de los 5 similares y la tabla detallada.
- **Pestaña 3 (Métricas):** Consulta los coeficientes del modelo y las gráficas diagnósticas de dispersión y tendencia.
