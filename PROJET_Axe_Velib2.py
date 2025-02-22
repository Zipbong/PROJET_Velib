
import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import MarkerCluster
from folium.plugins import HeatMap
# Import des librairies
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from folium.features import GeoJson, GeoJsonPopup, GeoJsonTooltip
import branca.colormap as cm
import os


# Nettoyage des cartes du session_state
map_keys = ["m_capa1", 'heatmapcapacityvelib', 'map_poi', 'map_surface', 'map_pop']
for map_key in map_keys:
    if map_key in st.session_state:
        del st.session_state[map_key]

# Import des fichiers avec décoration (cache)
@st.cache_data
def load_data_csv(filepath):
    data = pd.read_csv(filepath, sep=';')
    return data

@st.cache_data
def load_data_gpd(filepath):
    data = gpd.read_file(filepath)
    return data

@st.cache_data
def load_data_json(filepath):
    data = pd.read_json(filepath)
    return data

# Chemins relatifs vers les fichiers
chemin_csv = os.path.join(os.path.dirname(__file__), 'data', 'ParisDonneesSurfacePop.csv')
chemin_json = os.path.join(os.path.dirname(__file__), 'data', 'VelibEmplacementDesStations.json')
chemin_geojson = os.path.join(os.path.dirname(__file__), 'data', 'arrondissements.geojson')

# Lecture des fichiers
Paris_ref = load_data_csv(chemin_csv)
dfvelib = load_data_json(chemin_json)
arrondissements = load_data_gpd(chemin_geojson)

# Afficher les premières lignes des DataFrames pour vérifier
st.write(Paris_ref.head())
st.write(dfvelib.head())
st.write(arrondissements.head())







####################################################################################################################################################################
#### DEBUT AFFICHAGE
####################################################################################################################################################################
st.header("Axe capacité des stations Vélib'", divider = 'green')

st.markdown('''<p style='text-align: justify'>
                Nous allons dans cette section analyser les capacités des stations Vélib' de Paris intra-muros.
                </p>''',  unsafe_allow_html=True)
st.markdown('''<p style='text-align: justify'>
                Notre objectif est de pouvoir juger de la répartiiton des possibilités de locations de vélos au sein de la ville.
                </p>''',  unsafe_allow_html=True)



st.header('Présentation du fichier de reference', divider='green')
############################################################################
#### TRAITEMENT DU FICHIER DE REFERENCE POUR ETRE SANS LES SURFACES DES BOIS
st.markdown('''<p style='text-align: justify'>
                Nous utiliserons aussi un fichier contenant les surfaces et la population des arrondissements de Paris. 
                Il nous permettra de calculer des densités par arrondissement.</p>''',  unsafe_allow_html=True)
            
st.markdown('''<p style='text-align: justify'>
                De plus, concernant la surface, nous ne prendrons pas en compte la surface des bois de Boulogne et de Vincennes.
                En effet, ces deux zones sont très peu urbanisées et ne sont pas représentatives de la densité de population.
                D'autre part, elles augmentent fortement la surface de certains arrondissements et faussent les calculs de densité.
                </p>''',  unsafe_allow_html=True)

# Sans les zones boisées
Paris_ref.Nro = Paris_ref.Nro.replace({'12S' : 12, '16S' : 16})
# Créer un masque booléen pour identifier les lignes à supprimer
mask2 = Paris_ref['Nro'].isin(['12A', '16A'])
# Supprimer les lignes correspondantes au masque
Paris_ref_SB = Paris_ref[~mask2]
# Changement de type
Paris_ref_SB.Nro = Paris_ref_SB.Nro.astype('int')

checkbox1 = st.checkbox("Afficher les 5 premières lignes du fichier de référence")
if checkbox1:
    st.write(Paris_ref_SB.head())                                          



#############################################################################
st.header('Présentation du fichier de capacité des stations Vélib', divider='green')

st.markdown('''<p style='text-align: justify'>
                Le fichier dfvelib contient le nom, la capacité et la localisation des stations de Vélib'
                    </p>''',  unsafe_allow_html=True)
#### TRAITEMENT DE DFVELIB
# Affichage head avec selectbox   
checkboxvelib = st.checkbox("Afficher les 5 premières lignes du fichier de capacité Velib'")
if checkboxvelib:
    st.write(dfvelib.head())   
  

st.markdown('''<p style='text-align: justify'>
                Le fichier necessite un traitement pour être exploitable :
                </p>''',  unsafe_allow_html=True)
st.markdown('''<p style='text-align: justify'>
                - traitement des latitudes et longitudes,
                </p>''',  unsafe_allow_html=True)
st.markdown('''<p style='text-align: justify'>
                - ajout des arrondissements à partir du fichier arrondissements.geojson.
                </p>''',  unsafe_allow_html=True)

# Changement de type de stationcode
dfvelib['stationcode']=dfvelib['stationcode'].astype(str)

# Créer les colonnes 'longitude' et 'latitude'
dfvelib['longitude'] = dfvelib['coordonnees_geo'].apply(lambda x: x['lon'])
dfvelib['latitude'] = dfvelib['coordonnees_geo'].apply(lambda x: x['lat'])

# Ajout des arrondissements

# Create a GeoDataFrame from dfvelib
gdf_velib = gpd.GeoDataFrame(
    dfvelib, geometry=gpd.points_from_xy(dfvelib.longitude, dfvelib.latitude)
)

# Set the coordinate reference system (CRS) for both GeoDataFrames
gdf_velib = gdf_velib.set_crs(epsg=4326)  # Assuming your data is in WGS 84
arrondissements = arrondissements.to_crs(epsg=4326)  # Ensure both have the same CRS

# Perform a spatial join to find the arrondissement for each Velib station
joined_data = gpd.sjoin(gdf_velib, arrondissements, how="inner", predicate='intersects')

# Ajput et mise en forme des arrondissements
dfvelib['arrondissement'] = joined_data['c_ar']  # 
dfvelib['arrondissement'] = dfvelib['arrondissement'].astype(str)
dfvelib['arrondissement'] = dfvelib['arrondissement'].str.split('.').str[0]

# Supprimer les lignes avec arrondissement = 'nan' qui sont hors de Paris
mask2 = dfvelib['arrondissement'].isin(['nan'])
dfvelib_Paris = dfvelib[~mask2]

st.write(dfvelib_Paris.head())

st.header('Représentations graphiques des capacités des stations Vélib', divider = 'green')


 ############################################# CLUSTER #####################################
# Diviser la page en deux parties
col1, col2 = st.columns(2)

with col1:
    ##########################################################################
    # Creation de la carte cluster des capacité de velib
    if 'm_capa1' not in st.session_state: 
        # Coordonnées de Paris
        paris_coords = [48.8566, 2.3522]
        # Créer une carte centrée sur Paris
        m_capa1 = folium.Map(location=paris_coords, zoom_start=12, tiles="CartoDB dark_matter")

        #### CLUSTER

        # Créer un MarkerCluster
        marker_cluster = MarkerCluster().add_to(m_capa1)

        # Parcourir les données et ajouter un marqueur pour chaque aire de stationnement
        for index, row in dfvelib_Paris.iterrows():
            # Ajuster la taille du marqueur en fonction de la capacité
            radius = row['capacity'] / 10  # Ajuster le facteur de division si nécessaire

            folium.Marker(
                location=[row['latitude'], row['longitude']], # Use the 'Latitude' and 'Longitude' columns
                radius=radius,
                color='blue',  # Ou une autre couleur de votre choix
                fill=True,
                fill_color='blue',  # Ou une autre couleur de votre choix
                fill_opacity=0.6,
                popup=f"Station: {row['name']}<br>Capacity: {row['capacity']}"
            ).add_to(marker_cluster)  # Add to marker_cluster instead of m

        # Definition du style pour rendre les arrond plus clairs (LOWER fillOpacity to make heatmap visible)
        def style_function(feature):
            return {
                'fillOpacity': 0.2,  # Lower this to make heatmap more visible (try 0 if needed)
                'weight': 2,         # borders weight to separate Paris
                'color': 'black',
                'fillColor': 'white'  # Dark grey fill to contrast with map
            }

        # Ajout des arrond
        folium.GeoJson(
            arrondissements,
            name="Paris Arrondissements",
            style_function=style_function
        ).add_to(m_capa1)

        # Ajouter un titre à la carte
        title_html = '''
                <h3 align="center" style="font-size:20px">
                <b>Localisation des 'nan' dans le df considéré</b></h3>
                '''
        m_capa1.get_root().html.add_child(folium.Element(title_html))

        # Enregistrement de la carte dans l'état de la session
        st.session_state.m_capa1 = m_capa1


    # Afficher la carte dans Streamlit

    # Afficher la carte dans un conteneur avec des dimensions définies
    with st.container():
            st.write("**Cluster des lieux de stationnements vélo à Paris**")
            folium_map_sites_comptage_paris = st_folium(
                st.session_state.m_capa1, 
                width=800, 
                height=600,
                key="m_capa1",
                returned_objects=['last_active_drawing'])

with col2:
    st.markdown('''<p style='text-align: justify'>
                 <br><br><br><br><br><br>               
                </p>''',  unsafe_allow_html=True)
    st.markdown('''<p style='text-align: justify'>
                La carte ci-contre permet de visualiser la localisation des stations Vélib' à Paris.
                </p>''',  unsafe_allow_html=True)
    st.markdown('''<p style='text-align: justify'>
                On note une plus forte concentration de stations dans le centre de Paris (arrondissements 1 à 10), seul le 7eme parait moins équipé.
                Le 15eme arrondissement ressort aussi comme étant équipé en stations.
                </p>''',  unsafe_allow_html=True)







##########################################################################
 ########################### HEATMAP CAPACITE DES STATIONS VELIB #####################################
# Diviser la page en deux parties
col1, col2 = st.columns(2)

with col1 :
    st.markdown('''<p style='text-align: justify'>
                 <br><br><br><br><br><br>               
                </p>''',  unsafe_allow_html=True)
    st.markdown('''<p style='text-align: justify'>
                La carte ci-contre permet de visualiser les stations Vélib' à Paris.
                </p>''',  unsafe_allow_html=True)
    st.markdown('''<p style='text-align: justify'>
                Cette carte met clairement en exergue les arrondissements les mieux pourvus en stations.
                Ainsi, les arrondissements centraux sont mieux équipés que les arrondissements périphériques.
                <br><br>
                Les tailles des cercles nous donne une idée de la capacité des stations.
                </p>''',  unsafe_allow_html=True)
                

with col2:
    if 'heatmapcapacityvelib' not in st.session_state: 
        # Nécessaire pour travailler avec Folium, il faut avoir créer les colonnes latitude et longitude
        heat_data = [[row['latitude'], row['longitude']] for index, row in dfvelib_Paris.iterrows()]

        # Création de la Map
        heatmapcapacityvelib = folium.Map(location=[48.8566, 2.3522], zoom_start=13)

        # Ajouter un masque pour rendre les zones extérieures sombres
        folium.TileLayer('CartoDB dark_matter', opacity=0.6).add_to(heatmapcapacityvelib)

        # Ajout de la heatmap à la Map
        HeatMap(heat_data,
            min_opacity = 0.2,  # Set transparency (lower = more transparent)
            radius=20, # Size of the radius (larger = more spread out)
            blue = 15, # Blur of the points (larger = more diffuse)
            name = "Capacité de stations vélib dans Paris").add_to(heatmapcapacityvelib)

        # Function qui va nous permettre de customiser le layer 'arrondissement' (optionnel, dans mon example: basée sur la colonne 'c_ar')
        def style_function(feature):
            return {
                'fillOpacity': 0.4,     # Opacity of the fill
                'weight': 1,            # Line thickness
                'color': 'black',       # Color of the boundary line
                'fillColor': f'#{hex(int(feature["properties"]["c_ar"] * 1000))[2:8]}'  # Dynamic color based on 'c_ar'
            }

        # Ajouter le layer arrondissement comment un GeoJson layer
        folium.GeoJson(
            arrondissements,
            name="Arrondissements",
            style_function=style_function
        ).add_to(heatmapcapacityvelib)

        # Ajouter les noms des arrondissements comme annotations sur la carte
        for _, row in arrondissements.iterrows():
            centroid = row.geometry.centroid
            folium.Marker(
                location=[centroid.y, centroid.x],
                icon=folium.DivIcon(
                    html=f"""<div style="font-size: 10px; color : white; font-weight: bold; text-align: center;">
                        {row['l_ar']}</div>"""
                )
            ).add_to(heatmapcapacityvelib)

        # Créer les CircleMarkers avec des tailles adaptées
        fg4 = folium.FeatureGroup(name="Stations vélib").add_to(heatmapcapacityvelib)

        dfvelib_Paris['capacity_fixed'] = dfvelib_Paris['capacity'].clip(lower=1)  # Fixe un min de 1 pour éviter les cercles trop grands/petits

        dfvelib_Paris.apply(lambda row: folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=row['capacity_fixed'] / 10,  # Rayon proportionnel, minimum 1
        tooltip=f"Nom du site : {row['name']}<br>Capacity : {row['capacity']:.2f}",
        color='black',
        fill=True,
        weight=1,
        fill_color='white',
        fill_opacity=0.2
    ).add_to(fg4), axis=1)
        fg4.add_to(heatmapcapacityvelib)



        # Titre
        title_html = '''
                    <h3 align="center" style="font-size:20px">
                    <b>Stations Vélib' par arrondissement</b></h3>
                    '''

        heatmapcapacityvelib.get_root().html.add_child(folium.Element(title_html))

        # Ajout de LayerControl
        folium.LayerControl().add_to(heatmapcapacityvelib)

        # Enregistrement de la carte dans l'état de la session
        st.session_state.heatmapcapacityvelib = heatmapcapacityvelib


    # Afficher la carte dans Streamlit

    # Afficher la carte dans un conteneur avec des dimensions définies
    with st.container():
            st.write("**Heatmap des stations Vélib'**")
            folium_map_sites_comptage_paris = st_folium(
                st.session_state.heatmapcapacityvelib, 
                width=800, 
                height=600,
                key="heatmapcapacityvelib",
                returned_objects=['last_active_drawing'])

 


st.markdown('''<p style='text-align: justify'>
                Les deux précédentes cartes permettent de constater que le centre de Paris est mieux pourvu 
                que les arrondissements périphériques en terme de stations.
            <br>
                Cependant, nous allons désormais visualiser differemment ce constat et considerer les capacités.
                </p>''',  unsafe_allow_html=True)





########################################################################################################################
########################################################################################################################
### Calcul capacité par arrondissement et merge avec df de ref PAris_ref_SB
# Calculer la capacité totale par arrondissement
dfvelib_capacité_Paris = dfvelib_Paris.groupby('arrondissement')['capacity'].sum().reset_index()
dfvelib_capacité_Paris.reset_index()

# Changement de type
dfvelib_capacité_Paris['arrondissement'] = dfvelib_capacité_Paris['arrondissement'].astype(int)
# Fusionner les DataFrames en utilisant merge()
velib_capacité_merged_SB_df = pd.merge(dfvelib_capacité_Paris, Paris_ref_SB, left_on='arrondissement', right_on='Nro', how='left')
arrondissements['c_ar'] = arrondissements['c_ar'].astype(int)
velib_capacité_merged_SB_df['arrondissement'] = velib_capacité_merged_SB_df['arrondissement'].astype(int)




#########################################################################
st.subheader("Capacité par arrondissement")
#########################################################################
# Diviser la page en deux parties
col1, col2 = st.columns(2)
with col1:
    ### Carte choropleth - capacité par arrondissement
    plt.figure(figsize=(15,15))

    # Create a map centered on Paris
    map3 = folium.Map(location=[48.8566, 2.3522], zoom_start=12)

    # Ajouter un masque pour rendre les zones extérieures sombres
    folium.TileLayer('CartoDB dark_matter', opacity=0.6).add_to(map3)

    # Fusion des données de places_merge avec le GeoJson
    arrond_places = arrondissements.merge(velib_capacité_merged_SB_df, left_on='c_ar', right_on='arrondissement')

    # Créer la Choropleth map
    choropleth = folium.Choropleth(
        geo_data=arrondissements,
        name='carte choropleth',
        data=arrond_places,
        columns=["c_arinsee","capacity"],
        key_on="feature.properties.c_arinsee",
        fill_color='YlGn',
        weight = 1,
        color = 'white',
        fill_opacity=0.7,
        line_opacity=0.4,
        legend_name="Indicateurs",
        highlight=True
    ).add_to(map3)

    # Ajout des info-bulles (tooltips)
    tooltip = GeoJsonTooltip(
        fields=["c_arinsee",'capacity'],
        aliases=['Code Arrondissement', "Capacité des stations Velib"],
        localize=True
    )

    # Ajouter la couche GeoJson avec les info-bulles à la carte
    folium.GeoJson(
        data=arrond_places,
        name='Données arrondissement',
        tooltip=tooltip,
        style_function=lambda x: {
            'color': 'white',
            'weight': 1,
            'fillOpacity': 0.1,
            'lineColor': 'white'
        }
    ).add_to(map3)

    # Ajouter les noms des arrondissements comme annotations sur la carte
    for _, row in arrondissements.iterrows():
        centroid = row.geometry.centroid
        folium.Marker(
            location=[centroid.y, centroid.x],
            icon=folium.DivIcon(
                html=f"""<div style="font-size: 10px; font-color: white; font-weight: bold; text-align: center;">
                        {row['l_ar']}</div>"""
            )
        ).add_to(map3)

    # Add a title to the map
    title_html = '''
                <h3 align="center" style="font-size:20px">
                <b>Capacité de places Velib par arrondissement</h3>
                '''
    map3.get_root().html.add_child(folium.Element(title_html))


    # Ajouter une légende interactive
    folium.LayerControl().add_to(map3)


    # Afficher la carte avec streamlit
    st_folium(map3, width=700, height=500)

with col2:
    st.markdown('''<p style='text-align: justify'>
                 <br><br><br><br><br><br>               
                </p>''',  unsafe_allow_html=True)
    st.markdown('''<p style='text-align: justify'>
                La carte ci-contre permet de visualiser la capacité des stations Vélib' à Paris.
                </p>''',  unsafe_allow_html=True)
    st.markdown('''<p style='text-align: justify'>
                Considerer la capacité plutot que le nombre de stations Vélib apporte un éclairage different.
                Sous cet angle, les arrondissements centraux sont moins pourvus que les arrondissements périphériques.
                Le 15eme arrondissement est clairement celui qui a la plus forte capacité de Vélib.
                </p>''',  unsafe_allow_html=True)


#########################################################################
##### Histogramme choropleth capacite par arrondissement
# Changement de type pour permettre le merge
velib_capacité_merged_SB_df['arrondissement'] = velib_capacité_merged_SB_df['arrondissement'].astype(str)
# Trie des données en fonction xxx de manière décroissante
data_capacite_triee = velib_capacité_merged_SB_df.sort_values('capacity', ascending=False)
data_capacite_triee['arrondissement'] = data_capacite_triee['arrondissement'].astype(int)

# Définir les paliers de couleurs et la carte de couleurs dans l'ordre inverse
color_scale = [(0, '#ffffe5'), (0.2, '#e5f4ab'), (0.4, '#a2d889'), (0.6, '#4bb062'), (0.8, '#15783e'), (1, '#004529')]
color_midpoint = (data_capacite_triee['capacity'].max() + data_capacite_triee['capacity'].min()) / 2

# Création du graphique interactif avec plotly
fig = px.bar(data_capacite_triee,
             x=data_capacite_triee['arrondissement'],
             y='capacity',
             title="Capacité des stations Vélib",
             color='capacity',
             color_continuous_scale=color_scale,
             #range_y=[00, 40]
             )
# Mettre le fond du graphique en gris
fig.update_layout(plot_bgcolor='grey',
                  xaxis_type='category', # pour trier 
                  coloraxis_colorbar=dict(
                      title='Capacité',
                      tickvals=[
                          data_capacite_triee['capacity'].min(),
                          data_capacite_triee['capacity'].quantile(0.2),
                          data_capacite_triee['capacity'].quantile(0.4),
                          data_capacite_triee['capacity'].quantile(0.6),
                          data_capacite_triee['capacity'].quantile(0.8),
                          data_capacite_triee['capacity'].max()
                      ],
                      ticktext=['Très basse', 'Basse', 'Moyenne', 'Élevée', 'Très élevée', 'Maximum']
                  ))

# Afficher le graphique avec streamlit
st.plotly_chart(fig)





#######################################################################################################################################################
st.header('Analyse de densité',  divider = 'green')
#######################################################################################################################################################
#########################################################################
st.markdown('''<p style='text-align: justify'>
                Nous allons désormais utiliser le fichier de référence presenté précedemment.
                </p>''',  unsafe_allow_html=True)
st.markdown('''<p style='text-align: justify'>
                L'objectif est de visualiser par arrondissement : 
                </p>''',  unsafe_allow_html=True)
st.markdown('''<p style='text-align: justify'>
                - Capacité Velib' rapportée à la surface de l'arrondissement
                </p>''',  unsafe_allow_html=True)
st.markdown('''<p style='text-align: justify'>
                - Capacité Velib' rapportée à la population de l'arrondissement
                </p>''',  unsafe_allow_html=True)


### Traitement de du df merge pour obtenir des densites capacité / surface et capacité / population

# Calcul ratio
velib_capacité_merged_SB_df['Capacité par ha'] = velib_capacité_merged_SB_df['capacity'] / velib_capacité_merged_SB_df["Superficie_(ha)"]
velib_capacité_merged_SB_df['Capacité par 1000 hab'] = velib_capacité_merged_SB_df['capacity'] / velib_capacité_merged_SB_df["Population_(2020)"]

# Changement de type (sinon erreur dans le graphique)
velib_capacité_merged_SB_df['arrondissement'] = velib_capacité_merged_SB_df['arrondissement'].astype(str)

# Trie des données en fonction xxx de manière décroissante
data_capacite_triee = velib_capacité_merged_SB_df.sort_values('Capacité par ha', ascending=False)

# Définir les paliers de couleurs et la carte de couleurs dans l'ordre inverse
color_scale = [(0, '#ffffe5'), (0.2, '#e5f4ab'), (0.4, '#a2d889'), (0.6, '#4bb062'), (0.8, '#15783e'), (1, '#004529')]
color_midpoint = (data_capacite_triee['Capacité par ha'].max() + data_capacite_triee['Capacité par ha'].min()) / 2


#########################################################################
st.subheader("Capacité selon la surface par arrondissement")
#########################################################################

    ### Graphe histogramme  capacité / surface
    # Création du graphique interactif avec plotly
fig = px.bar(data_capacite_triee,
                x=data_capacite_triee['arrondissement'],
                y="Capacité par ha",
                title="Capacité par hectare",
                color='Capacité par ha',
                color_continuous_scale=color_scale,
                #range_y=[00, 40]
                )

    # Ajouter les annotations au survol de la souris
    #fig.update_traces(hovertemplate="<b>Arrondissement : %{x}</b><br>Part du commerce, transports et services divers, en % : %{y}")

    # Mettre le fond du graphique en gris
fig.update_layout(plot_bgcolor='grey',
                    xaxis_type='category', # pour trier 
                    coloraxis_colorbar=dict(
                        title='Capacité par ha',
                            tickvals=[
                            data_capacite_triee['Capacité par ha'].min(),
                            data_capacite_triee['Capacité par ha'].quantile(0.2),
                            data_capacite_triee['Capacité par ha'].quantile(0.4),
                            data_capacite_triee['Capacité par ha'].quantile(0.6),
                            data_capacite_triee['Capacité par ha'].quantile(0.8),
                            data_capacite_triee['Capacité par ha'].max()
                        ],
                        ticktext=['Très basse', 'Basse', 'Moyenne', 'Élevée', 'Très élevée', 'Maximum']
                    ))

    # Afficher le graphique
st.plotly_chart(fig)



# Diviser la page en deux parties
col1, col2 = st.columns(2)

with col1:
        #########################################################################
        ### Carte choropleth  capacité / surface
        # Création du graphique interactif avec plotly

    arrondissements['c_ar'] = arrondissements['c_ar'].astype(int)
    velib_capacité_merged_SB_df['arrondissement'] = velib_capacité_merged_SB_df['arrondissement'].astype(int)

    plt.figure(figsize=(10,10))

    # Create a map centered on Paris
    map4 = folium.Map(location=[48.8566, 2.3522], zoom_start=12)

    # Ajouter un masque pour rendre les zones extérieures sombres
    folium.TileLayer('CartoDB dark_matter', opacity=0.6).add_to(map3)

    # Fusion des données de places_merge avec le GeoJson
    arrond_places = arrondissements.merge(velib_capacité_merged_SB_df, left_on='c_ar', right_on='arrondissement')
    # arrond_places.info()

    # Créer la Choropleth map
    choropleth = folium.Choropleth(
            geo_data=arrondissements,
            name='carte choropleth',
            data=arrond_places,
            columns=["c_arinsee","Capacité par ha"],
            key_on="feature.properties.c_arinsee",
            fill_color='YlGn',
            weight = 1,
            color = 'white',
            fill_opacity=0.7,
            line_opacity=0.4,
            legend_name="Indicateurs",
            highlight=True
        ).add_to(map4)

    # Ajout des info-bulles (tooltips)
    tooltip = GeoJsonTooltip(
            fields=["c_arinsee",'Capacité par ha'],
            aliases=['Code Arrondissement', "Capacité des stations Velib"],
            localize=True
        )

    # Ajouter la couche GeoJson avec les info-bulles à la carte
    folium.GeoJson(
            data=arrond_places,
            name='Données arrondissement',
            tooltip=tooltip,
            style_function=lambda x: {
                'color': 'white',
                'weight': 1,
                'fillOpacity': 0.1,
                'lineColor': 'white'
            }
        ).add_to(map4)

    # Ajouter les noms des arrondissements comme annotations sur la carte
    for _, row in arrondissements.iterrows():
            centroid = row.geometry.centroid
            folium.Marker(
                location=[centroid.y, centroid.x],
                icon=folium.DivIcon(
                    html=f"""<div style="font-size: 10px; font-color: white; font-weight: bold; text-align: center;">
                            {row['l_ar']}</div>"""
                )
            ).add_to(map4)

    # Add a title to the map
    title_html = '''
                    <h3 align="center" style="font-size:20px">
                    <b>Capacité de places Velib selon la surface par arrondissement</h3>
                    '''
    map4.get_root().html.add_child(folium.Element(title_html))

    # Ajouter une légende interactive
    folium.LayerControl().add_to(map4)

    # Afficher la carte avec streamlit
    st_folium(map4, width=700, height=500)

with col2:
    st.markdown('''<p style='text-align: justify'>
                 <br><br><br><br><br><br>               
                </p>''',  unsafe_allow_html=True)
    st.markdown('''<p style='text-align: justify'>
                La carte ci-contre et l'histogramme au-dessus permettent de visualiser la capacité des stations Vélib' à Paris.
                </p>''',  unsafe_allow_html=True)
    st.markdown('''<p style='text-align: justify'>
                Considerer la capacité par hectare permet de mettre en avant les arrondissements les mieux pourvus en stations  Vélib.
                Ainsi, les arrondissements centraux sont mieux équipés que les arrondissements périphériques. 
                <br>
                Relativement à sa surface, le 2eme est le mieux loti.
                </p>''',  unsafe_allow_html=True)



#########################################################################
st.subheader("Capacité selon la population par arrondissement")
#########################################################################




### Graphe histogramme  capacité / population
# Création du graphique interactif avec plotly

velib_capacité_merged_SB_df['arrondissement'] = velib_capacité_merged_SB_df['arrondissement'].astype(str)
# Trie des données en fonction xxx de manière décroissante
data_capacite_triee = velib_capacité_merged_SB_df.sort_values('Capacité par 1000 hab', ascending=False)

# Définir les paliers de couleurs et la carte de couleurs dans l'ordre inverse
color_scale = [(0, '#ffffe5'), (0.2, '#e5f4ab'), (0.4, '#a2d889'), (0.6, '#4bb062'), (0.8, '#15783e'), (1, '#004529')]
color_midpoint = (data_capacite_triee['Capacité par 1000 hab'].max() + data_capacite_triee['Capacité par 1000 hab'].min()) / 2

# Création du graphique interactif avec plotly
fig = px.bar(data_capacite_triee,
             x=data_capacite_triee['arrondissement'],
             y='Capacité par 1000 hab',
             title='Capacité par 1000 hab',
             color='Capacité par 1000 hab',
             color_continuous_scale=color_scale,
             #range_y=[00, 40]
             )

# Ajouter les annotations au survol de la souris
#fig.update_traces(hovertemplate="<b>Arrondissement : %{x}</b><br>Part du commerce, transports et services divers, en % : %{y}")

# Mettre le fond du graphique en gris
fig.update_layout(plot_bgcolor='grey',
                  xaxis_type='category', # pour trier 
                  coloraxis_colorbar=dict(
                      title='Capacité par 1000 hab',
                      tickvals=[
                          data_capacite_triee['Capacité par 1000 hab'].min(),
                          data_capacite_triee['Capacité par 1000 hab'].quantile(0.2),
                          data_capacite_triee['Capacité par 1000 hab'].quantile(0.4),
                          data_capacite_triee['Capacité par 1000 hab'].quantile(0.6),
                          data_capacite_triee['Capacité par 1000 hab'].quantile(0.8),
                          data_capacite_triee['Capacité par 1000 hab'].max()
                      ],
                      ticktext=['Très basse', 'Basse', 'Moyenne', 'Élevée', 'Très élevée', 'Maximum']
                  ))

# Afficher le graphique
st.plotly_chart(fig)



# Diviser la page en deux parties
col1, col2 = st.columns(2)

with col1:
    #########################################################################
    ### Carte choropleth  capacité / population
    # Création du graphique interactif avec plotly

    arrondissements['c_ar'] = arrondissements['c_ar'].astype(int)
    velib_capacité_merged_SB_df['arrondissement'] = velib_capacité_merged_SB_df['arrondissement'].astype(int)

    plt.figure(figsize=(10,10))

    # Create a map centered on Paris
    map4 = folium.Map(location=[48.8566, 2.3522], zoom_start=12)

    # Ajouter un masque pour rendre les zones extérieures sombres
    folium.TileLayer('CartoDB dark_matter', opacity=0.6).add_to(map3)

    # Fusion des données de places_merge avec le GeoJson
    arrond_places = arrondissements.merge(velib_capacité_merged_SB_df, left_on='c_ar', right_on='arrondissement')

    # Créer la Choropleth map
    choropleth = folium.Choropleth(
        geo_data=arrondissements,
        name='carte choropleth',
        data=arrond_places,
        columns=["c_arinsee","Capacité par 1000 hab"],
        key_on="feature.properties.c_arinsee",
        fill_color='YlGn',
        weight = 1,
        color = 'white',
        fill_opacity=0.7,
        line_opacity=0.4,
        legend_name="Indicateurs",
        highlight=True
    ).add_to(map4)

    # Ajout des info-bulles (tooltips)
    tooltip = GeoJsonTooltip(
        fields=["c_arinsee",'Capacité par 1000 hab'],
        aliases=['Code Arrondissement', "Capacité des stations Velib"],
        localize=True
    )

    # Ajouter la couche GeoJson avec les info-bulles à la carte
    folium.GeoJson(
        data=arrond_places,
        name='Données arrondissement',
        tooltip=tooltip,
        style_function=lambda x: {
            'color': 'white',
            'weight': 1,
            'fillOpacity': 0.1,
            'lineColor': 'white'
        }
    ).add_to(map4)

    # Ajouter les noms des arrondissements comme annotations sur la carte
    for _, row in arrondissements.iterrows():
        centroid = row.geometry.centroid
        folium.Marker(
            location=[centroid.y, centroid.x],
            icon=folium.DivIcon(
                html=f"""<div style="font-size: 10px; font-color: white; font-weight: bold; text-align: center;">
                        {row['l_ar']}</div>"""
            )
        ).add_to(map4)

    # Add a title to the map
    title_html = '''
                <h3 align="center" style="font-size:20px">
                <b>Capacité de places Velib selon la population par arrondissement</h3>
                '''
    map4.get_root().html.add_child(folium.Element(title_html))

    # Ajouter une légende interactive
    folium.LayerControl().add_to(map4)

    # Display the map# Afficher la carte avec streamlit
    st_folium(map4, width=700, height=500)

with col2:
    st.markdown('''<p style='text-align: justify'>
                 <br><br><br><br><br><br>               
                </p>''',  unsafe_allow_html=True)   
    st.markdown('''<p style='text-align: justify'>
                La carte ci-contre et l'histogramme au-dessus permettent de visualiser la capacité des stations Vélib' à Paris rapportée à
                la population de chaque arrondissement.
                Sous cet angle, les arrondissements centraux sont mieux équipés que les arrondissements périphériques.
                <br>
                Relativement à leur population, le 8eme et le 1er sont les mieux lotis.
                </p>''',  unsafe_allow_html=True)
    

st.header('Conclusion', divider='green')
st.markdown('''<p style='text-align: justify'>
                Nous avons pu visualiser la localisation des stations Vélib' à Paris et leur capacité sour certains angles.
                <br>
                Selon chaque angle un arrondissement ou un autre ressort comme étant mieux pourvu en stations ou capacité.
                <br>
                Il est interessant de noter que le 1er arrondissement avec la plus petite population de Paris (15 000 habitants) ressorte comme fortement doté en Vélib. 
                Cependant, cet arrondissement est aussi et surtout un point d'interet touristique majeur, accueillant chaque année des millions de visiteurs.
                Un axe d'amelioration de notre étude aurait pu être d'analyse la capacité des stations Vélib' en fonction de la fréquentation touristique.
                <br><br>
                La conclusion de notre étude est que les arrondissements centraux sont mieux pourvus en stations Vélib' que les arrondissements périphériques.        
                L'implantation des Velib' étant multifactorielle, il est difficile de conclure sur les raisons de cette disparité avec seulement l'etude de la surface et de la population    
                </p>''',  unsafe_allow_html=True)
