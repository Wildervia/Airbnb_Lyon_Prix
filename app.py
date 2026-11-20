from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

BASE = Path(__file__).resolve().parent
st.set_page_config(page_title='Lyon | Rental Intelligence',page_icon='🏙️',layout='wide',initial_sidebar_state='expanded')
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"], [data-testid="stApp"] {font-family:'DM Sans',sans-serif;}
.stApp{background:linear-gradient(135deg,#f5f9ff 0%,#fffaf5 100%);color:#193450}
.block-container{padding-top:1.45rem;padding-bottom:3rem;max-width:1450px}
section[data-testid="stSidebar"]{background:#eaf3fc;border-right:1px solid #dbe8f5}
section[data-testid="stSidebar"] *{color:#203e5b}
[data-testid="stMetric"]{background:#fff;border:1px solid #e0eaf5;border-radius:20px;padding:19px;box-shadow:0 8px 24px rgba(36,74,117,.055)}
[data-testid="stMetricValue"]{color:#1b5c96;font-weight:800}
h1,h2,h3{color:#183a5c;letter-spacing:-.035em}
.stTabs [data-baseweb="tab-list"]{gap:14px;border-bottom:1px solid #e2eaf3}
.stTabs [data-baseweb="tab"]{border-radius:12px 12px 0 0;font-weight:700;padding:10px 20px}
.stTabs [aria-selected="true"]{color:#e98240!important;border-bottom:3px solid #e98240!important}
div[data-testid="stPlotlyChart"]{background:#fff;border:1px solid #e2eaf3;border-radius:20px;padding:8px;box-shadow:0 8px 24px rgba(36,74,117,.04)}
.stButton button,.stDownloadButton button{background:#e98240;color:white;border:0;border-radius:12px}
.hero{background:linear-gradient(110deg,#1d5d96,#4b92c9);padding:28px 32px;border-radius:24px;color:white;margin-bottom:22px;box-shadow:0 12px 30px rgba(27,92,150,.14)}
.hero h1{color:white;font-size:2.2rem;margin:0 0 8px 0}.hero p{color:#e9f3ff;margin:0}
.section-note{color:#68849c;font-size:.92rem}
</style>""",unsafe_allow_html=True)

@st.cache_data(show_spinner='Chargement des données…')
def load_data():
    return pd.read_parquet(BASE / 'data' / 'airbnb_lyon.parquet')

@st.cache_data(show_spinner='Lecture des avis…')
def load_reviews():
    return pd.read_parquet(BASE / 'data' / 'airbnb_lyon_reviews.parquet')

def num(series):
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series,errors='coerce')
    s=series.astype(str).str.replace(r'[^\d,.-]','',regex=True).str.replace(',','.',regex=False)
    return pd.to_numeric(s,errors='coerce')

def euro(value):
    return f'{value:,.0f} €'.replace(',',' ') if pd.notna(value) else 'N/D'

st.markdown('<div class="hero"><h1>LYON · Rental Intelligence</h1><p>Le marché locatif en un regard · Analyse, carte et estimation intelligente</p></div>',unsafe_allow_html=True)
try:
    raw=load_data()
except Exception as exc:
    st.error(f'Lecture du fichier impossible : {exc}')
    st.info('Vérifiez que data/airbnb_lyon.parquet existe et que pyarrow est installé.')
    st.stop()

required=['prix_nuit','quartier','room_type','accommodates']
missing=[c for c in required if c not in raw.columns]
if missing:
    st.error('Colonnes manquantes : '+', '.join(missing))
    st.write('Colonnes disponibles :',list(raw.columns))
    st.stop()

df=raw.copy()
for c in ['prix_nuit','accommodates','bedrooms','surface','surface_m2','superficie','square_meters','square_feet','latitude','longitude','estimated_occupancy_l365d','estimated_revenue_l365d','review_scores_rating','number_of_reviews','availability_365']:
    if c in df: df[c]=num(df[c])
df=df[df.prix_nuit.notna() & (df.prix_nuit > 0)].copy()
if df.empty:
    st.error('Aucune annonce ne possède un prix par nuit valide.')
    st.stop()

with st.sidebar:
    st.header('🎛️ Explorer le marché')
    areas=sorted(df.quartier.dropna().astype(str).unique())
    selected_areas=st.multiselect('Quartiers',areas,placeholder='Tous les quartiers')
    types=sorted(df.room_type.dropna().astype(str).unique())
    selected_types=st.multiselect('Type de logement',types,placeholder='Tous les types')
    maxcap=max(1,min(20,int(df.accommodates.max())))
    mincap=st.slider('Capacité minimum',1,maxcap,1)
    if 'bedrooms' in df and df.bedrooms.notna().any():
        maxbed=max(0,min(10,int(df.bedrooms.max())))
        minbed=st.slider('Nombre minimum de chambres',0,maxbed,0)
    else: minbed=0
    low=int(df.prix_nuit.min());high=int(df.prix_nuit.max())
    price_range=st.slider('Prix par nuit (€)',low,high,(low,high)) if low<high else (low,high)
    st.divider()
    st.caption('Filtres appliqués au marché et à la carte. La prédiction utilise les annonces du dataset complet.')

filtered=df.copy()
if selected_areas:filtered=filtered[filtered.quartier.astype(str).isin(selected_areas)]
if selected_types:filtered=filtered[filtered.room_type.astype(str).isin(selected_types)]
filtered=filtered[(filtered.accommodates>=mincap)&filtered.prix_nuit.between(*price_range)]
if 'bedrooms' in filtered:filtered=filtered[filtered.bedrooms.fillna(0)>=minbed]
if filtered.empty:
    st.sidebar.warning('Aucune annonce ne correspond aux filtres du marché. La prédiction reste disponible.')

market,estimate,maptab=st.tabs(['📈 Vue du marché','🏡 Estimer mon bien','🗺️ Carte interactive'])

BLUE='#3c82ba'
ORANGE='#ef9555'
COLORS=['#2876BC','#F28B35','#19A59B','#9362C9']
TYPE_COLORS={'Entire home/apt':'#2876BC','Private room':'#F28B35','Hotel room':'#19A59B','Shared room':'#9362C9'}
def type_colors(types):
    return {name: TYPE_COLORS.get(name, '#65758B') for name in types}
def chart_style(fig, title=None):
    fig.update_layout(template='plotly_white',font=dict(family='DM Sans, sans-serif',color='#31536f'),
        paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=22,r=22,t=58,b=28),title=dict(text=title or '',font=dict(size=18,color='#21466b')),
        colorway=COLORS,hoverlabel=dict(bgcolor='#ffffff',font_size=13))
    fig.update_xaxes(showgrid=True,gridcolor='#eaf0f6',zeroline=False)
    fig.update_yaxes(showgrid=True,gridcolor='#eaf0f6',zeroline=False)
    return fig

def occupancy_pct(values):
    values=num(values).dropna()
    values=values[(values>=0)&(values<=365)]
    if values.empty:return values
    # l365d : nombre de jours sur 365 ; les proportions (<=1) sont aussi gérées.
    if values.max()<=1:return values*100
    if values.max()<=100:return values # échelle en pourcentage lorsque cohérente
    return values/365*100

with market:
    st.markdown('### Panorama du marché lyonnais')
    st.caption('Les indicateurs se mettent à jour avec les filtres de gauche · Données Airbnb fournies')
    if filtered.empty:
        st.warning('Aucune annonce avec les filtres choisis. Élargissez les critères à gauche.')
    else:
        k1,k2,k3,k4=st.columns(4)
        k1.metric('Prix moyen / nuit',euro(filtered.prix_nuit.mean()))
        k2.metric('Prix médian / nuit',euro(filtered.prix_nuit.median()))
        k3.metric('Annonces analysées',f'{len(filtered):,}'.replace(',',' '))
        if 'estimated_occupancy_l365d' in filtered:
            oc=occupancy_pct(filtered['estimated_occupancy_l365d'])
            k4.metric('Occupation estimée médiane',f'{oc.median():.0f} %' if len(oc) else 'N/D')
        else:k4.metric('Occupation estimée médiane','N/D')
        st.markdown('#### Comprendre les prix')
        left,right=st.columns(2)
        with left:
            # Ne pas transformer une date de collecte ou de commentaire en historique de prix.
            dated_candidates=['date_prix','price_date','date_observation_prix','observation_date','snapshot_date','date_snapshot']
            date_col=next((c for c in dated_candidates if c in filtered.columns),None)
            if date_col:
                temporal=filtered[['prix_nuit',date_col]].copy()
                temporal['annee']=pd.to_datetime(temporal[date_col],errors='coerce').dt.year
                temporal=temporal.dropna(subset=['annee'])
                grouped=temporal.groupby('annee').agg(prix_moyen=('prix_nuit','mean'),annonces=('prix_nuit','size')).reset_index()
                if len(grouped)>1:
                    fig=px.line(grouped,x='annee',y='prix_moyen',markers=True,custom_data=['annonces'])
                    fig.update_traces(line_color=BLUE,line_width=3,marker=dict(size=9,color=ORANGE),hovertemplate='Année : %{x}<br>Prix moyen : %{y:.0f} €<br>Annonces : %{customdata[0]}<extra></extra>')
                    st.plotly_chart(chart_style(fig,'Année / prix moyen'),use_container_width=True)
                # Pas de graphique si les observations annuelles sont insuffisantes.
            # Ne pas afficher de message technique en l'absence d'historique de prix.
            distribution=filtered.copy()
            upper=distribution.prix_nuit.quantile(.99)
            distribution=distribution[distribution.prix_nuit<=upper]
            fig=px.histogram(distribution,x='prix_nuit',nbins=42,color_discrete_sequence=[BLUE])
            fig.update_layout(xaxis_title='Prix par nuit (€)',yaxis_title='Nombre d’annonces')
            st.plotly_chart(chart_style(fig,'Distribution des prix par nuit'),use_container_width=True)
        with right:
            surface_col=next((c for c in ['surface_m2','surface','superficie','square_meters','square_feet'] if c in filtered and num(filtered[c]).notna().any()),None)
            if surface_col:
                frame=filtered.copy();frame['_surface']=num(frame[surface_col])
                if surface_col=='square_feet':frame['_surface']=frame['_surface']*0.092903
                frame=frame[(frame['_surface']>5)&(frame['_surface']<600)&(frame.prix_nuit<=frame.prix_nuit.quantile(.99))]
                if not frame.empty:
                    fig=px.scatter(frame,x='_surface',y='prix_nuit',color='room_type' if 'room_type' in frame else None,
                                   opacity=.52,color_discrete_sequence=COLORS,hover_data=['quartier'] if 'quartier' in frame else None)
                    fig.update_layout(xaxis_title='Surface (m²)',yaxis_title='Prix / nuit (€)',legend_title_text='Type')
                    st.plotly_chart(chart_style(fig,'Surface et prix par nuit'),use_container_width=True)
                # Ne rien afficher si la surface n'est pas exploitable.
            # Ne pas afficher de message technique si la surface est absente.
            if 'estimated_occupancy_l365d' in filtered:
                frame=filtered.copy();frame['_occ']=occupancy_pct(frame['estimated_occupancy_l365d']).reindex(frame.index)
                frame=frame.dropna(subset=['_occ','prix_nuit'])
                frame=frame[frame.prix_nuit<=frame.prix_nuit.quantile(.99)] if len(frame) else frame
                if len(frame):
                    fig=px.scatter(frame,x='prix_nuit',y='_occ',opacity=.48,color_discrete_sequence=[ORANGE],
                                   hover_data=['quartier'] if 'quartier' in frame else None)
                    fig.update_layout(xaxis_title='Prix par nuit (€)',yaxis_title='Occupation estimée (%)')
                    st.plotly_chart(chart_style(fig,'Prix et taux d’occupation estimé'),use_container_width=True)
                else:st.info('Prix et occupation : pas de valeurs exploitables.')
            else:st.info('Prix et occupation : champ d’occupation absent du dataset.')
        st.markdown('#### Géographie et structure de l’offre')
        left,right=st.columns(2)
        with left:
            grouped=filtered.groupby('quartier',dropna=True).agg(prix=('prix_nuit','mean'),annonces=('prix_nuit','size')).query('annonces >= 3').sort_values('prix').tail(12).reset_index()
            if len(grouped):
                fig=px.bar(grouped,x='prix',y='quartier',orientation='h',color_discrete_sequence=[BLUE],custom_data=['annonces'])
                fig.update_traces(hovertemplate='%{y}<br>Prix moyen : %{x:.0f} €<br>Annonces : %{customdata[0]}<extra></extra>')
                fig.update_layout(xaxis_title='Prix moyen / nuit (€)',yaxis_title='')
                st.plotly_chart(chart_style(fig,'Prix moyen par quartier'),use_container_width=True)
        with right:
            counts=filtered.room_type.fillna('Non renseigné').value_counts().reset_index()
            counts.columns=['type','annonces']
            counts['part']=100*counts['annonces']/counts['annonces'].sum()
            counts['legende']=counts.apply(lambda r:f"{r['type']} · {r['annonces']:,} ({r['part']:.2f} %)",axis=1)
            fig=px.pie(counts,names='legende',values='annonces',hole=.56,
                       color='type',color_discrete_map=type_colors(counts['type']))
            fig.update_traces(textinfo='none',pull=0,sort=False,
                              marker=dict(line=dict(color='#FFFFFF',width=0)),
                              hovertemplate='%{label}<extra></extra>')
            fig.update_layout(legend=dict(orientation='h',y=-0.18,x=0,font=dict(size=11)),
                              margin=dict(l=12,r=12,t=60,b=105),height=430)
            st.plotly_chart(chart_style(fig,'Répartition par type de logement'),use_container_width=True)
            st.caption('Les pourcentages et effectifs de toutes les catégories figurent dans la légende.')

        st.markdown('#### Prix et ancienneté des annonces')
        if 'first_review' in filtered.columns:
            cohorts=filtered[['first_review','room_type','prix_nuit']].copy()
            cohorts['annee']=pd.to_datetime(cohorts['first_review'],errors='coerce').dt.year
            cohorts=cohorts.dropna(subset=['annee','prix_nuit','room_type'])
            cohorts=cohorts[cohorts['annee'].between(2008, pd.Timestamp.today().year)]
            cohort_group=cohorts.groupby(['annee','room_type']).agg(prix_moyen=('prix_nuit','mean'),annonces=('prix_nuit','size')).reset_index()
            cohort_group=cohort_group[cohort_group['annonces']>=3]
            if not cohort_group.empty and cohort_group.annee.nunique()>1:
                fig=px.line(cohort_group,x='annee',y='prix_moyen',color='room_type',markers=True,
                            color_discrete_map=type_colors(cohort_group.room_type.unique()),custom_data=['annonces'])
                fig.update_traces(hovertemplate='Année du premier avis : %{x}<br>Prix actuel moyen : %{y:.0f} €<br>Annonces : %{customdata[0]}<extra></extra>')
                years=sorted(cohort_group['annee'].astype(int).unique().tolist())
                fig.update_xaxes(tickmode='array',tickvals=years,dtick=1)
                fig.update_layout(xaxis_title='Année du premier avis',yaxis_title='Prix actuel moyen / nuit (€)',legend_title_text='Type de chambre')
                st.plotly_chart(chart_style(fig,'Prix actuel selon l’année du premier avis et le type'),use_container_width=True)
                st.caption('Comparaison des cohortes d’annonces par année de leur premier avis. Ce graphique ne représente PAS une évolution historique des prix : le dataset ne contient pas les prix pratiqués chaque année.')
            else: st.info('Données insuffisantes pour comparer les cohortes par type de chambre.')
        else: st.info('Année du premier avis indisponible : aucune comparaison annuelle des cohortes possible.')

        st.markdown('#### Avis des voyageurs : quels types de chambres sont les plus appréciés ?')
        try:
            reviews=load_reviews()
            needed={'listing_id','date'}
            if not needed.issubset(reviews.columns) or 'id_annonce' not in filtered.columns:
                st.warning('Impossible de relier les avis aux annonces : identifiants ou dates manquants.')
            else:
                reviews=reviews.copy()
                reviews['listing_id']=pd.to_numeric(reviews['listing_id'],errors='coerce')
                reviews['date']=pd.to_datetime(reviews['date'],errors='coerce')
                reviews=reviews.dropna(subset=['listing_id','date'])
                listings=filtered[['id_annonce','room_type']].drop_duplicates(subset=['id_annonce']).copy()
                listings['id_annonce']=pd.to_numeric(listings['id_annonce'],errors='coerce')
                joined=reviews.merge(listings,left_on='listing_id',right_on='id_annonce',how='inner',validate='many_to_one')
                if joined.empty:
                    st.info('Aucun avis lié aux annonces sélectionnées. Élargissez les filtres.')
                else:
                    a,b,c=st.columns(3)
                    a.metric('Avis reliés aux annonces',f'{len(joined):,}'.replace(',',' '))
                    b.metric('Logements avec avis',f'{joined.listing_id.nunique():,}'.replace(',',' '))
                    c.metric('Types de chambres évalués',str(joined.room_type.nunique()))
                    joined['annee']=joined['date'].dt.year
                    annual=joined.groupby(['annee','room_type']).size().reset_index(name='avis')
                    fig=px.bar(annual,x='annee',y='avis',color='room_type',barmode='stack',
                               color_discrete_map=type_colors(annual.room_type.unique()))
                    years=sorted(annual['annee'].astype(int).unique().tolist())
                    fig.update_xaxes(tickmode='array',tickvals=years)
                    fig.update_layout(xaxis_title='Année de publication de l’avis',yaxis_title='Nombre d’avis',legend_title_text='Type de chambre')
                    st.plotly_chart(chart_style(fig,'Évolution annuelle du volume des avis par type'),use_container_width=True)
                    c1,c2=st.columns(2)
                    with c1:
                        counts=joined.groupby('room_type').size().reset_index(name='avis').sort_values('avis',ascending=True)
                        fig=px.bar(counts,x='avis',y='room_type',orientation='h',color='room_type',
                                   color_discrete_map=type_colors(counts.room_type.unique()),text='avis')
                        fig.update_layout(showlegend=False,xaxis_title='Nombre d’avis',yaxis_title='')
                        st.plotly_chart(chart_style(fig,'Nombre d’avis par type de chambre'),use_container_width=True)
                    with c2:
                        if 'review_scores_rating' in filtered.columns:
                            rating=filtered[['id_annonce','room_type','review_scores_rating']].copy()
                            rating['review_scores_rating']=num(rating['review_scores_rating'])
                            rating=rating.dropna(subset=['review_scores_rating'])
                            # La note provient de la fiche annonce, pas des commentaires individuels.
                            rating=rating.groupby('room_type').agg(note=('review_scores_rating','mean'),logements=('review_scores_rating','size')).reset_index()
                            rating=rating[rating.logements>=3].sort_values('note')
                            if not rating.empty:
                                fig=px.bar(rating,x='note',y='room_type',orientation='h',color='room_type',
                                           color_discrete_map=type_colors(rating.room_type.unique()),
                                           custom_data=['logements'],text='note')
                                fig.update_traces(texttemplate='%{x:.2f}',textposition='outside',
                                                  hovertemplate='%{y}<br>Note moyenne : %{x:.2f}<br>Logements notés : %{customdata[0]}<extra></extra>')
                                fig.update_layout(showlegend=False,xaxis_title='Note moyenne des annonces',yaxis_title='')
                                st.plotly_chart(chart_style(fig,'Types de chambres les mieux notés'),use_container_width=True)
                                best=rating.iloc[-1]
                                st.success(f"Type le mieux noté parmi les logements filtrés : {best['room_type']} ({best['note']:.2f} en moyenne sur {int(best['logements'])} annonces notées).")
                            else: st.info('Pas assez de notes disponibles pour comparer les types.')
                        else: st.info('Aucune note chiffrée dans les fiches annonces.')
                    st.caption('Les avis datés proviennent du fichier airbnb_lyon_reviews.parquet, relié aux annonces par listing_id = id_annonce. Les notes sont celles des annonces dans airbnb_lyon.parquet. Le volume d’avis mesure l’activité, pas la satisfaction ; aucune analyse de sentiment des commentaires n’est réalisée.')
        except Exception as exc:
            st.error(f'Impossible de charger ou relier le fichier des avis : {exc}')

@st.cache_resource(show_spinner='Entraînement du modèle de prédiction…')
def train_price_model(data):
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    categorical = [c for c in ['quartier', 'room_type', 'property_type'] if c in data]
    numerical = [c for c in ['accommodates', 'bedrooms', 'beds', 'bathrooms'] if c in data]
    # Éviter les variables qui révèlent indirectement le prix ou ne sont pas connues du propriétaire.
    features = categorical + numerical
    source = data.loc[data.prix_nuit.notna() & (data.prix_nuit > 0), features + ['prix_nuit']].copy()
    for c in categorical:
        source[c] = source[c].fillna('Inconnu').astype(str)
    for c in numerical:
        source[c] = num(source[c])
    source = source.dropna(subset=['prix_nuit'])
    if len(source) < 30:
        raise ValueError('Moins de 30 annonces valides pour entraîner un modèle.')
    x_train, x_test, y_train, y_test = train_test_split(
        source[features], np.log1p(source['prix_nuit']), test_size=.2, random_state=42
    )
    prep = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
        ('num', SimpleImputer(strategy='median'), numerical),
    ])
    model = Pipeline([
        ('prep', prep),
        ('reg', ExtraTreesRegressor(n_estimators=140, min_samples_leaf=3,
                                    max_features=.9, random_state=42, n_jobs=-1))
    ])
    model.fit(x_train, y_train)
    actual = np.expm1(y_test)
    predicted = np.expm1(model.predict(x_test))
    mae = float(mean_absolute_error(actual, predicted))
    r2 = float(r2_score(actual, predicted))
    errors = np.abs(actual - predicted)
    error80 = float(np.quantile(errors, .8))
    return model, features, categorical, numerical, mae, r2, error80, len(source)

with estimate:
    st.subheader('Quel est le potentiel locatif de mon bien ?')
    st.caption('Estimation fondée sur les annonces du dataset lyonnais, même sans correspondance exacte.')
    try:
        model, features, categorical, numerical, mae, r2, error80, n_train = train_price_model(df)
    except Exception as exc:
        st.error(f'Impossible d’entraîner le modèle : {exc}')
    else:
        a,b,c,d=st.columns(4)
        with a: area=st.selectbox('Quartier du bien', areas)
        with b: room=st.selectbox('Type de logement', types, index=(types.index('Entire home/apt') if 'Entire home/apt' in types else 0))
        with c: cap=st.number_input('Voyageurs',min_value=1,max_value=20,value=2)
        with d: bedrooms=st.number_input('Chambres',min_value=0,max_value=12,value=1)
        values={'quartier':area,'room_type':room,'accommodates':cap,'bedrooms':bedrooms}
        if 'property_type' in features:
            choices=sorted(df.property_type.dropna().astype(str).unique())
            default=next((v for v in choices if 'Apartment' in v), choices[0]) if choices else 'Inconnu'
            values['property_type']=st.selectbox('Catégorie du bien',choices,index=choices.index(default)) if choices else 'Inconnu'
        x,y=st.columns(2)
        with x:
            if 'beds' in features: values['beds']=st.number_input('Nombre de lits',min_value=0,max_value=30,value=max(1,int(cap)))
        with y:
            if 'bathrooms' in features: values['bathrooms']=st.number_input('Salles de bain',min_value=0.,max_value=15.,value=1.,step=.5)
        sample=pd.DataFrame([{c:values.get(c,np.nan) for c in features}])
        price=float(np.expm1(model.predict(sample)[0]))
        price=max(0,price)
        low=max(0,price-error80)
        high=price+error80
        st.markdown('#### Estimation du prix par nuit')
        c1,c2,c3=st.columns(3)
        c1.metric('Prix prédit / nuit',euro(price))
        c2.metric('Fourchette indicative basse',euro(low))
        c3.metric('Fourchette indicative haute',euro(high))
        st.caption('Fourchette construite avec le 80e percentile des erreurs absolues sur le jeu de test. Ce n’est pas un intervalle de confiance garanti pour chaque logement.')
        st.markdown('#### Projection des revenus bruts')
        if 'estimated_occupancy_l365d' in df.columns:
            occupancy_source = df.loc[(df['quartier'].astype(str) == str(area)) & (df['room_type'].astype(str) == str(room)), 'estimated_occupancy_l365d'].dropna()
            occupancy_scope = 'même quartier et même type'
            if occupancy_source.empty:
                occupancy_source = df.loc[df['room_type'].astype(str) == str(room), 'estimated_occupancy_l365d'].dropna()
                occupancy_scope = 'même type à Lyon'
            if occupancy_source.empty:
                occupancy_source = df['estimated_occupancy_l365d'].dropna()
                occupancy_scope = 'ensemble des annonces à Lyon'
            if not occupancy_source.empty:
                occupancy_source = occupancy_source[(occupancy_source >= 0) & (occupancy_source <= 365)]
            if not occupancy_source.empty:
                occ_median = float(occupancy_source.median())
                # Les valeurs peuvent être des proportions, des pourcentages ou des jours occupés.
                if occupancy_source.max() <= 1: occ = occ_median * 100
                elif occupancy_source.max() <= 100: occ = occ_median
                else: occ = occ_median / 365 * 100
                occ = float(np.clip(occ, 0, 100))
                st.metric('Occupation estimée issue du dataset', f'{occ:.1f} %')
                st.caption(f'Médiane de {len(occupancy_source)} annonces : {occupancy_scope}. Champ source : estimated_occupancy_l365d. Pas une occupation observée garantie.')
            else:
                occ = None
        else:
            occ = None
        if occ is None:
            st.warning('Le dataset ne contient pas de taux d’occupation exploitable : aucune projection de revenus ne sera calculée.')
        nights=365/12*occ/100 if occ is not None else None
        if nights is not None:
            c1,c2,c3=st.columns(3)
            c1.metric('Revenu mensuel central',euro(price*nights))
            c2.metric('Revenu annuel central',euro(price*365*occ/100))
            c3.metric('Fourchette mensuelle',f'{euro(low*nights)} – {euro(high*nights)}')
        st.info('Montants bruts théoriques, hors commissions, charges, taxes, ménage et contraintes réglementaires. Le prix affiché dans les annonces n’est pas nécessairement le prix réellement payé.')
        st.markdown('#### Performance du modèle')
        c1,c2,c3=st.columns(3)
        c1.metric('Erreur moyenne absolue (MAE)',euro(mae))
        c2.metric('R² sur données de test',f'{r2:.2f}')
        c3.metric('Annonces disponibles',f'{n_train:,}'.replace(',',' '))
        st.caption('Évaluation sur 20 % des annonces non utilisées pour entraîner le modèle. Le score ne garantit pas la précision pour un logement atypique.')
        support=df[(df.quartier.astype(str)==str(area)) & (df.room_type.astype(str)==str(room))]
        if len(support)<10:
            st.warning(f'Attention : seulement {len(support)} annonces de ce type dans ce quartier. La prédiction repose surtout sur des logements d’autres secteurs ou catégories ; prudence.')
        else:
            st.caption(f'{len(support)} annonces de même type dans ce quartier pour contextualiser la prédiction.')
        with st.expander('Voir des annonces du même quartier et type'):
            cols=[c for c in ['titre','quartier','room_type','accommodates','bedrooms','prix_nuit'] if c in support]
            st.dataframe(support[cols].head(50),use_container_width=True,hide_index=True)

with maptab:
    st.subheader('📍 Répartition géographique des annonces')
    st.caption('Survolez un point pour voir les informations du logement. Zoomez et déplacez la carte librement.')
    if {'latitude', 'longitude'}.issubset(filtered.columns):
        geo = filtered.dropna(subset=['latitude', 'longitude']).copy()
        geo = geo[geo.latitude.between(45.5, 46) & geo.longitude.between(4.5, 5.2)].copy()
        if geo.empty:
            st.info('Aucune coordonnée exploitable pour les filtres choisis.')
        else:
            # Préparer des champs lisibles sans supposer que les colonnes facultatives existent.
            def field_text(frame, column, default='Non renseigné'):
                if column not in frame.columns:
                    return pd.Series(default, index=frame.index, dtype='object')
                return frame[column].fillna(default).astype(str)

            geo['_titre_carte'] = field_text(geo, 'titre', 'Annonce Airbnb').str.slice(0, 85)
            geo['_quartier_carte'] = field_text(geo, 'quartier')
            geo['_type_carte'] = field_text(geo, 'room_type')
            geo['_prix_carte'] = geo.prix_nuit.apply(euro)
            geo['_capacite_carte'] = geo.accommodates.apply(lambda v: f'{int(v)} voyageur(s)' if pd.notna(v) else 'Non renseigné')
            if 'bedrooms' in geo.columns:
                geo['_chambres_carte'] = geo.bedrooms.apply(lambda v: str(int(v)) if pd.notna(v) else 'Non renseigné')
            else:
                geo['_chambres_carte'] = 'Non renseigné'
            if 'review_scores_rating' in geo.columns:
                geo['_note_carte'] = geo.review_scores_rating.apply(lambda v: f'{v:.1f}' if pd.notna(v) else 'Non renseignée')
            else:
                geo['_note_carte'] = 'Non renseignée'
            if 'number_of_reviews' in geo.columns:
                geo['_avis_carte'] = geo.number_of_reviews.apply(lambda v: str(int(v)) if pd.notna(v) else 'Non renseigné')
            else:
                geo['_avis_carte'] = 'Non renseigné'

            fields = ['_titre_carte', '_quartier_carte', '_type_carte', '_prix_carte',
                      '_capacite_carte', '_chambres_carte', '_note_carte', '_avis_carte']
            fig = px.scatter_mapbox(
                geo, lat='latitude', lon='longitude',
                custom_data=fields,
                zoom=11, center={'lat': float(geo.latitude.median()), 'lon': float(geo.longitude.median())},
                height=630, mapbox_style='open-street-map',
            )
            fig.update_traces(
                marker={'size': 9, 'color': '#E95B53', 'opacity': 0.72},
                hovertemplate=(
                    '<b>%{customdata[0]}</b><br>'
                    'Quartier : %{customdata[1]}<br>'
                    'Type : %{customdata[2]}<br>'
                    '<b>Prix / nuit : %{customdata[3]}</b><br>'
                    'Capacité : %{customdata[4]}<br>'
                    'Chambres : %{customdata[5]}<br>'
                    'Note : %{customdata[6]}<br>'
                    'Nombre d’avis : %{customdata[7]}'
                    '<extra></extra>'
                )
            )
            fig.update_layout(margin={'l': 0, 'r': 0, 't': 0, 'b': 0}, showlegend=False)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
            st.caption(f'{len(geo):,} annonces géolocalisées • Les informations manquantes sont signalées dans les infobulles.'.replace(',', ' '))
    else:
        st.info('Coordonnées géographiques indisponibles.')
