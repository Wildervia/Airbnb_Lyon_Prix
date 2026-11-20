import pandas as pd
import streamlit as st

st.set_page_config(page_title="Prix Airbnb Lyon", layout="wide")


@st.cache_data
def charger():
    df = pd.read_parquet("data/airbnb_lyon.parquet")
    df["arrondissement"] = df["quartier"].str.replace(" Arrondissement", "")
    return df


df = charger()

# 1. La barre latérale : le logement du propriétaire
TYPES = {"Logement entier": "Entire home/apt", "Chambre privée": "Private room"}

st.sidebar.header("Mon logement")
arr = st.sidebar.selectbox("Arrondissement", sorted(df["arrondissement"].unique()))
type_choisi = st.sidebar.radio("Type de logement", list(TYPES))
voyageurs = st.sidebar.slider("Voyageurs", 1, 8, 4)

# 2. Les annonces comparables : même arrondissement, même type, même capacité, avec un prix
comp = df[
    (df["arrondissement"] == arr)
    & (df["room_type"] == TYPES[type_choisi])
    & (df["accommodates"] == voyageurs)
    & (df["prix_nuit"].notna())
]

# 3. Le titre et les KPI
st.title("À quel prix louer mon logement à Lyon ?")
st.caption("Source : Inside Airbnb, relevé du 19/09/2026")

if comp.empty:
    st.warning("Aucune annonce comparable : changez l'arrondissement, le type ou la capacité.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Prix médian / nuit", f"{comp['prix_nuit'].median():.0f}")
c2.metric(
    "Fourchette du marché",
    f"{comp['prix_nuit'].quantile(0.25):.0f} – {comp['prix_nuit'].quantile(0.75):.0f}",
)
c3.metric("Occupation estimée", f"{comp['estimated_occupancy_l365d'].median():.0f} nuits/an")
c4.metric("Annonces comparables", len(comp))

st.caption("Prix tels qu'écrits dans la source (symbole « $ », devise non précisée).")