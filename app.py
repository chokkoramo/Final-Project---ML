"""
Aplicación Principal: Tasador y Buscador de Vehículos Usados.
Desarrollada con Streamlit.
"""

import streamlit as st

def main():
    st.set_page_config(
        page_title="Tasador y Buscador de Vehículos Usados",
        page_icon="🚗",
        layout="wide",
    )
    st.title("🚗 Tasador y Buscador de Vehículos Usados")
    st.info("Inicializando aplicación...")

if __name__ == "__main__":
    main()
