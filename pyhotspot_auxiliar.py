import numpy as np
import networkx as nx
import pandas as pd
import geopandas as gpd
import osmnx as ox
from math import radians, cos, sin, asin, sqrt, floor, pi
from scipy.spatial import cKDTree
from sklearn.neighbors import BallTree

import folium
from folium.plugins import MarkerCluster, Fullscreen
from copy import deepcopy


def quartic_gaussian_distance(distances, bandwidth):
    return np.where(distances <= bandwidth, 15. / 16 * (1 - (distances / bandwidth) ** 2) ** 2, 0)

def degree_per_meter(latitude):
    """
    Approximates the value of one meter to degree 
    according to the latitude of the region
    
    Parameters
    ----------
    latitude : float
    Region average latitude
    
    Returns
    -------
    meters : float
    The proportion of degree of 1 meter
    """
    rlat = float(latitude) * pi / 180
    # meter per degree Latitude
    meters_lat = 111132.92 - 559.82 * cos(2 * rlat) + 1.175 * cos(4 * rlat)
    # meter per degree Longitude
    meters_lgn = 111412.84 * cos(rlat) - 93.5 * cos(3 * rlat)
    meters = (meters_lat + meters_lgn) / 2
    return meters

def multigraph_to_graph(multigraph, weight='length', min_weight=True):
    """
    Transform multigraph to graph and keep the minimum or maximum weight.
    
    Parameters
    ----------
    multigraph : networkx multigraph
    Networkx multigraph.
    
    weight : string or None, (default='length')
    The edge attribute that holds the numerical value used as a weight. 
    If None, then each edge has weight 1.
    
    Returns
    -------
    G : Networkx Graph
    """

    G = nx.Graph()
    G.graph.update(deepcopy(multigraph.graph))
    G.add_nodes_from((n, deepcopy(d)) for n, d in multigraph._node.items())
    
    for u, nbrs in multigraph._adj.items():
        for v, keydict in nbrs.items():
            for key, data in keydict.items():
                if key == 0:
                    add_data = data
                else:
                    w = data[weight] if weight in data else 1.0
                    if min_weight:
                        if w < add_data[weight]:
                            add_data = data
                    else:
                        if w > add_data[weight]:
                            add_data = data
            G.add_edge(u, v, **deepcopy(add_data))
    
    return G

def nearest_edges_index(gdf_edges, gdf_events, method='balltree', dist=0.0001,
                        return_distance=False):
    """
    Return the index of edges nearest to the events.
    
    Parameters
    ----------
    gdf_edges : GeoDataFrame
        Edges dataframe with geometry LineString
    gdf_events : GeoDataFrame
        Events dataframe with geometry Point
    method : string, default='kdtree', {'kdtree', 'balltree'}
        Which method to use for finding nearest edge to each edge.
        If 'kdtree' we use scipy.spatial.cKDTree for very fast euclidean search. 
        Recommended for projected graphs. If 'balltree', we use 
        sklearn.neighbors.BallTree for fast haversine search. Recommended 
        for unprojected graphs.
    dist : float
        spacing length along edges. Units are the same as the geom; Degrees for
        unprojected geometries and meters for projected geometries. The smaller
        the value, the more points are created.
    return_distance : bool
        if True, return one array with the distance of the edge to each 
        closest event
    
    Returns
    -------
    eidx : ndarray
        Vector with the indexes of each closest edge
    dist : ndarray (optional)
        Vector with the distance of each closest edge
        
    Notes
    -----
    This function is adapted from get_nearest_edges developed by 
    samuelduchesne on osmnx
    """
    
    X = gdf_events.geometry.x
    Y = gdf_events.geometry.y
    
    # transform edges into evenly spaced points
    gdf_edges['points'] = gdf_edges.apply(
        lambda x: ox.utils_geo.redistribute_vertices(x.geometry, dist), axis=1
    )

    # develop edges data for each created points
    extended = (
        gdf_edges['points']
        .apply([pd.Series])
        .stack()
        .reset_index(level=1, drop=True)
        .join(gdf_edges)
        .reset_index()
    )
    # drop points information from dataframe
    gdf_edges.drop('points', axis=1, inplace=True)
    
    if method == 'kdtree':
        
        # check if we were able to import scipy.spatial.cKDTree successfully
        if not cKDTree:
            raise ImportError("The scipy package must be installed to use this optional feature.")
            
        # Prepare btree arrays
        nbdata = np.array(
            list(
                zip(
                    extended['Series'].apply(lambda x: x.x),
                    extended['Series'].apply(lambda x: x.y)
                )
            )
        )

        # build a k-d tree for euclidean nearest node search
        btree = cKDTree(nbdata)
        # query the tree for nearest node to each point
        points = np.array([X, Y]).T
        dist, idx = btree.query(points, k=1)  # Returns ids of closest point
        eidx = extended.loc[idx, 'index'].values
        
    elif method == "balltree":
        
        # check if we were able to import sklearn.neighbors.BallTree successfully
        if not BallTree:
            raise ImportError(
                "The scikit-learn package must be installed to use this optional feature."
            )
            
        # haversine requires data in form of [lat, lng] and inputs/outputs in units of radians
        nodes = pd.DataFrame(
            {
                "x": extended["Series"].apply(lambda x: x.x),
                "y": extended["Series"].apply(lambda x: x.y),
            }
        )
        nodes_rad = np.deg2rad(nodes[["y", "x"]].values.astype(np.float))
        points = np.array([Y, X]).T
        points_rad = np.deg2rad(points)

        # build a ball tree for haversine nearest node search
        tree = BallTree(nodes_rad, metric="haversine")

        # query the tree for nearest node to each point
        if not return_distance:
            idx = tree.query(points_rad, k=1, return_distance=False)
        else:
            dist, idx = tree.query(points_rad, k=1, return_distance=True)
            dist = np.rad2deg(dist).flatten()
        eidx = extended.loc[idx[:, 0], 'index'].values
    
    if return_distance: 
        return eidx, dist
    else:
        return eidx
    
    
def plot_area_folium(area_geo):
        
    study_area_geojson_poly = gpd.GeoSeries([area_geo]).to_json()
    study_area_plot = folium.features.GeoJson(study_area_geojson_poly, 
                                              style_function=lambda feature: {
            'color': 'black',
            'weight' : 1,
            'fillOpacity' : 0.0,
            'name': 'teste',
            })
    # Create the map object with focus on the center
    folium_map = folium.Map(tiles = 'cartodbpositron')
    # Create the two layers of feature group
    fg_boundary = folium.FeatureGroup(name='Boundary').add_to(folium_map)
    # Add the AIS exterior line to the map
    study_area_plot.add_to(fg_boundary)
    # Add full screen button
    Fullscreen().add_to(folium_map)
    # Changes the view of the map based on geometry
    sw = [area_geo.bounds[1], area_geo.bounds[0]]
    ne = [area_geo.bounds[3], area_geo.bounds[2]]
    folium_map.fit_bounds([sw, ne])

    return folium_map

def plot_network_events_folium(area_geo, plot_events, plot_edges, 
                               maximum_cluster=100):

    # Take the folium map
    folium_map = plot_area_folium(area_geo)

    #select_roads = select_roads.reset_index()

    # Create the two layers of feature group
    fg_cluster = folium.FeatureGroup(name='Clusters').add_to(folium_map)
    fg_roads = folium.FeatureGroup(name='Roads').add_to(folium_map)

    popup = folium.Popup()
    # Exclude events without nearest edge
    plot_events = plot_events[(plot_events['nearest_edge'].isin(plot_edges.index.tolist()))]

    area_geo_poly = folium.GeoJson(area_geo, style_function=lambda feature: {
        'fillColor': 'blue',
        'color' : 'blue',
        'weight' : 1,
        'fillOpacity' : 0.1,
        })
    popup.add_to(area_geo_poly)

    for rank, (index, edge) in enumerate(plot_edges.iterrows()):
        road_events = plot_events[plot_events['nearest_edge'] == index]

        # Converto to GeoJson
        geojson_poly = gpd.GeoSeries(edge['geometry']).to_json()

        popup = folium.Popup('Rank: '+str(rank+1)+' Crimes: '+str(len(road_events)))
        edge_plot = folium.features.GeoJson(geojson_poly, 
                                               style_function=lambda feature: {
            'color': 'red',
            'weight' : 5,
            'fillOpacity' : 0.0,
            })

        # Not create cluster above the maximum desired (avoid render problem)
        if rank < maximum_cluster:

            mc = MarkerCluster()

            for tmp_event in road_events.itertuples():
                folium.CircleMarker(
                    [tmp_event.geometry.y, tmp_event.geometry.x],
                    popup=tmp_event.geometry.x,
                    radius=2,
                    color='red',
                    fill=True,
                    fill_opacity=0.7,
                    ).add_to(mc)

            fg_cluster.add_child(mc)

        popup.add_to(edge_plot)
        fg_roads.add_child(edge_plot)

    # Add the select box of layers
    folium.LayerControl().add_to(folium_map)
    folium_map

    return folium_map