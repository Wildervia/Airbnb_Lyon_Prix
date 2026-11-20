import pandas as pd
import streamlit as st

st.set_page_config(page_title="Prix Airbnb Lyon", layout="wide")


@st.cache_data
def charger():
    return pd.read_parquet("data/airbnb_lyon.parquet")


df = charger()
st.title("À quel prix louer mon appartement à Lyon ?")
st.caption("Source : Inside Airbnb, relevé du 19/09/2026")
nb_prix = df["prix_nuit"].notna().sum()
st.metric("Annonces avec un prix", f"{nb_prix} sur {len(df)}")