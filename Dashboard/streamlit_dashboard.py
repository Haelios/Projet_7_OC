import streamlit as st
from streamlit_shap import st_shap
from streamlit_extras.altex import bar_chart
import altair as alt
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings
from shap import summary_plot, force_plot
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")


# API endpoint URL
API_URL = 'http://BalancerTest-1660040751.eu-west-3.elb.amazonaws.com/prediction'


# client_id = 100002 # (Pour ajouter des infos et debug, sans avoir à entrer d'ID à chaque fois)

options = st.sidebar.radio('Détails', options=['Accueil', 'Feature Importance', 'Comparaisons', 'Analyse Bivariée'])


def get_response():
    st.title('Retrouvez ici les résultats de votre demande de prêt')
    id_widget = st.empty()
    id_client = id_widget.number_input("Veuillez entrer votre identifiant.")
    if id_client :
        id_widget.empty()
        resp = requests.post(API_URL, json={'loan_id': id_client})
        return resp, id_client
    else:
        st.stop()


def accueil():
    """
    Page d'accueil du Dashboard, affiche le résultat de la demande de prêt, avec une jauge affichant le score (proba)
    du client et le seuil utilisé pour la prédiction.
    """
    # Vérifier que la requête a bien fonctionné
    if st.session_state.response.status_code == 200:
        # Vérifier que la réponse contient des données
        if 'error' in st.session_state.response.json().keys():
            st.write(st.session_state.response.json()['error'])
        else:
            index = st.session_state.response.json()['index']
            client_prediction = st.session_state.response.json()['prediction'][index]

            # Afficher le résultat de la demande
            st.subheader('Vos résultats :')
            if client_prediction == 0:
                st.write('Votre prêt a été accepté !')
            else:
                st.write("Votre demande de prêt n'a pas été acceptée.")

            # Afficher la jauge avec le score du client
            score = st.session_state.response.json()['score'][index]
            if score > 0.5 :
                color = 'red'
            else:
                color = 'green'

            fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = (1-score)*100,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Votre Score", 'font': {'size': 24}},
                            gauge = {
                                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                                'bar': {'color': color, 'thickness': 1},
                                'borderwidth': 2,
                                'bordercolor': "black",
                                'threshold': {
                                    'line': {'color': "black", 'width': 3},
                                    'thickness': 1,
                                    'value': 50}}))

            st.plotly_chart(fig)
            if score > 0.5 :
                st.write("Vous devez obtenir un score d'au moins 50 pour être éligible pour un prêt.")
    else:
        st.error('Error retrieving loan prediction. Please try again.')
        st.text(st.session_state.response.status_code)


def feature_imp():

    # Récupération des données via la requête
    all_data = pd.DataFrame(st.session_state.response.json()['all_client_val'], columns=st.session_state.response.json()['all_client_col'])
    shap_values = st.session_state.response.json()['shap_val']
    expected_value = st.session_state.response.json()['expected_val']

    # Création du summary plot de feature importance globale
    st.header("Les informations les plus importantes pour calculer votre score")
    st.image('Data/feature_imp_glob.png', width=900)

    # Création du force plot de feature importance locale
    fig3 = force_plot(expected_value, np.array(shap_values), all_data, feature_names=all_data.columns)
    st.header("Influence de ces données sur votre score")
    st_shap(fig3, width=900)


def comparison_graphs():

    ext_values = st.session_state.response.json()['comp_graphs'][0]
    ext_feature = ['Ext1', 'Ext2', 'Ext3', 'Ext1', 'Ext2', 'Ext3']
    ext_clients = ['client', 'client', 'client', 'mean', 'mean', 'mean']
    df_ext = pd.DataFrame([ext_values, ext_feature, ext_clients], index= ['values', 'feature', 'clients']).T.fillna(0)

    bar_chart(
        data=df_ext,
        x="clients:N",
        y="values:Q",
        color="clients:N",
        column="feature:N",
        title="External Sources : Comparaison entre vos données et les prêts accordés",
        width=150,
        use_container_width=False,
    )

    days_values = st.session_state.response.json()['comp_graphs'][1]
    days_feat = ['Days Employed']
    days_clients = ['client', 'mean']
    df_days = pd.DataFrame([days_values, days_feat, days_clients], index=['values', 'feature', 'clients']).T.fillna(0)
    
    bar_chart(
        data=df_days,
        x="clients",
        y="values",
        title="Days employed : Comparaison entre vos données et les prêts accordés",
        width=500,
        height=500,
        color = "clients"
    )


def bivar_graphs():
    index = st.session_state.response.json()['index']
    scores = st.session_state.response.json()['score']
    ext1 = st.session_state.response.json()['bivar_graphs'][0]
    ext2 = st.session_state.response.json()['bivar_graphs'][1]
    ext3 = st.session_state.response.json()['bivar_graphs'][2]
    target = st.session_state.response.json()['prediction']
    df_bivar = pd.DataFrame(zip(ext1, ext2, ext3, scores), columns = ['Ext1', 'Ext2', 'Ext3', 'Score'])

    st.header("Influence des sources extérieures sur les résultats")
    df_client = pd.DataFrame(zip([ext1[index]], [ext2[index]], [ext3[index]], [scores[index]]),
                                 columns = ['Ext1', 'Ext2', 'Ext3', 'Score']).fillna(0)

    # Ext1 vs Ext2
    fig1 = alt.Chart(df_bivar).mark_point(filled=True).encode(
        x="Ext1",
        y="Ext2",
        color="Score:Q").properties(
        width = 800,
        height=800,
        title="Graphique pour EXT SOURCE 1 et 2"
    )

    fig2 = alt.Chart(df_client).mark_circle(color='red', size=150).encode(
        x="Ext1",
        y="Ext2").properties(
        width=800,
        height=800
    )
    fig = fig1 + fig2

    st.altair_chart(fig)

    # Ext2 vs Ext3
    fig1 = alt.Chart(df_bivar).mark_point(filled=True).encode(
        x="Ext2",
        y="Ext3",
        color="Score:Q").properties(
        width=800,
        height=800,
        title="Graphique poour EXT SOURCE 2 et 3"
    )

    fig2 = alt.Chart(df_client).mark_circle(color='red', size=150).encode(
        x="Ext2",
        y="Ext3").properties(
        width=800,
        height=800
    )
    fig = fig1 + fig2

    st.altair_chart(fig)

    # Ext1 vs Ext3
    fig1 = alt.Chart(df_bivar).mark_point(filled=True).encode(
        x="Ext1",
        y="Ext3",
        color="Score:Q").properties(
        width=800,
        height=800,
        title="Graphique pour EXT SOURCE 1 et 3"
    )

    fig2 = alt.Chart(df_client).mark_circle(color='red', size=150).encode(
        x="Ext1",
        y="Ext3").properties(
        width=800,
        height=800
    )
    fig = fig1 + fig2

    st.altair_chart(fig)


if 'response' not in st.session_state:
    st.session_state.response, st.session_state.client_id = get_response()
if 'client_id' in st.session_state:
    st.sidebar.write(st.session_state.client_id)
    if options == 'Accueil':
        accueil()
    elif options == 'Feature Importance':
        feature_imp()
    elif options == 'Comparaisons':
        comparison_graphs()
    elif options == 'Analyse Bivariée':
        bivar_graphs()
