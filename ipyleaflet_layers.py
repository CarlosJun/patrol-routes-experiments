import ipyleaflet
import joblib
import geopandas as gpd
import base64
import ipywidgets as widgets
from shapely.geometry import Point
from mpl_toolkits.axes_grid1 import make_axes_locatable
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, MultiPoint, LineString, MultiLineString, Polygon, mapping
from io import BytesIO
import pandas as pd
import json
import re

def graph_to_gdfs(G, nodes=True, edges=True, node_geometry=True, fill_edge_geometry=True):
    """
    Convert a graph to node and/or edge GeoDataFrames.

    Parameters
    ----------
    G : networkx.Graph
        input graph
    nodes : bool
        if True, convert graph nodes to a GeoDataFrame and return it
    edges : bool
        if True, convert graph edges to a GeoDataFrame and return it
    node_geometry : bool
        if True, create a geometry column from node x and y data
    fill_edge_geometry : bool
        if True, fill in missing edge geometry fields using nodes u and v
    Returns
    -------
    geopandas.GeoDataFrame or tuple
        gdf_nodes or gdf_edges or tuple of (gdf_nodes, gdf_edges)
    
    Notes
    -----
    This function is adapted from graph_to_gdfs developed by osmnx
    """
    if not (nodes or edges):
        raise ValueError("You must request nodes or edges, or both.")

    crs = G.graph["crs"]
    to_return = []

    if nodes:

        nodes, data = zip(*G.nodes(data=True))

        if node_geometry:
            # convert node x/y attributes to Points for geometry column
            geom = (Point(d["x"], d["y"]) for d in data)
            gdf_nodes = gpd.GeoDataFrame(data, index=nodes, crs=crs, geometry=list(geom))
        else:
            gdf_nodes = gpd.GeoDataFrame(data, index=nodes, crs=crs)

        to_return.append(gdf_nodes)

    if edges:

        if len(G.edges()) < 1:
            raise ValueError("Graph has no edges, cannot convert to a GeoDataFrame.")

        if issubclass(type(G), nx.MultiGraph):
            u, v, k, data = zip(*G.edges(keys=True, data=True))
        else:
            u, v, data = zip(*G.edges(data=True))

        if fill_edge_geometry:

            # subroutine to get geometry for every edge: if edge already has
            # geometry return it, otherwise create it using the incident nodes
            x_lookup = nx.get_node_attributes(G, "x")
            y_lookup = nx.get_node_attributes(G, "y")

            def make_geom(u, v, data, x=x_lookup, y=y_lookup):
                if "geometry" in data:
                    return data["geometry"]
                else:
                    return LineString((Point((x[u], y[u])), Point((x[v], y[v]))))

            geom = map(make_geom, u, v, data)
            gdf_edges = gpd.GeoDataFrame(data, crs=crs, geometry=list(geom))

        else:
            gdf_edges = gpd.GeoDataFrame(data, crs=crs)
            # if no edges had a geometry attribute, create null column
            if "geometry" not in gdf_edges.columns:
                gdf_edges["geometry"] = np.nan
            gdf_edges.set_geometry("geometry")

        # add u, v, key attributes as columns
        gdf_edges["u"] = u
        gdf_edges["v"] = v
        if issubclass(type(G), nx.MultiGraph):
            gdf_edges["key"] = k

        to_return.append(gdf_edges)

    if len(to_return) > 1:
        return tuple(to_return)
    else:
        return to_return[0]

def round_trip_size(graph, weight='length'):
    
    if not isinstance(graph, nx.Graph):
        raise Exception("Expected nx.Graph, but {} was found".format(type(graph)))
    
    # get the minimum cycle basis
    cycle_basis = nx.minimum_cycle_basis(graph)
    cycle_basis_size = sum([nx.induced_subgraph(graph, cycle).size(weight=weight)
                            for cycle in cycle_basis])
    
    # edge that does not belong to any cycle count twice
    route_size = cycle_basis_size + 2*sum([graph[edge[0]][edge[1]][weight] 
                                           for edge in nx.bridges(graph)])
    
    return route_size

def matplot_datetime_heatmap(df_, scale_minx=0, scale_maxx=23, datetime_col='datetime', 
                 cmap='Reds', nan_as_zero=True, textcolors=("black", "white"), 
                 threshold=None, **textkw):
    """Creates matplotlib heatmap for couting events occurred by datetime
    
    Parameters
    ----------
    df_ : DataFrame of GeoDataFrame
        Dataframe to look at the datetime column
    scale_minx : int
        Initial time interval
    scale_maxx : int
        Ending time interval
    datetime_col : str
        Datetime column to look at on the dataset
    cmap : str
        Cmap colors of the heatmap, default is 'Reds'
    nan_as_zero : bool
        If should deal with nans putting zeros or not
    textcolos : list
        List of text colors to use on the plot
    
    Returns
    -------
    heatmap_plot : matplotlib plot
    
    Returns a heatmap made with matplotlib and based on the number of occurrencies of a datetime.
    """
    
    if scale_minx <= scale_maxx:
        scale_x = [i for i in range(scale_minx,scale_maxx+1)]
    else:
        scale_x = [i for i in range(scale_minx,24)] + [j for j in range(0,scale_maxx+1)]
    
    df = df_[[datetime_col]].copy()
    df.loc[:,'COUNT'] = 1
    df.loc[:, 'days'] = df_[datetime_col].apply(lambda day: day.dayofweek)
    df.loc[:, 'hour'] = df_[datetime_col].apply(lambda day: day.hour)
    df_grouped = df.groupby(by=['days', 'hour']).agg({'COUNT':'sum'}).reset_index()
    if nan_as_zero:
        arr = np.zeros((7,24))
    else:
        arr = np.nan * np.zeros((7,24))
    for i, row in df_grouped.iterrows():
        arr[int(row['days']), int(row['hour'])] = row['COUNT']
        
    arr = arr[:, scale_x]
        
    fig, ax = plt.subplots(figsize=(15,5))
    
    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)
        
    ax.set_yticks([i for i in range(7)])
    #ax.set_yticklabels(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    ax.set_yticklabels(['M', 'T', 'W', 'T', 'F', 'S', 'S'])
    
    ax.set_xticks([x for x in range(len(scale_x))]) #int
    ax.set_xticklabels([f"{x:02d}" for x in scale_x])
    plt.xticks(rotation=90)
    
    im = ax.imshow(arr, cmap=cmap, interpolation='none', aspect=1.5)
    plt.title('Occurrences by weekday and hour', fontsize=12)
    
    divider = make_axes_locatable(ax)
    cax = divider.new_vertical(size="5%", pad=0.5, pack_start=True, aspect=0.05)
    fig.add_axes(cax)
    fig.colorbar(im, cax=cax, orientation="horizontal")
    
    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(arr.max())/2.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)
    
    for (i, j), day in np.ndenumerate(arr):
        if np.isfinite(day):
            kw.update(color=textcolors[int(im.norm(arr[i, j]) > threshold)])
            ax.text(j, i, int(day), **kw)
    
    plt.close(fig)
    
    return fig

def heatmap_popup(events_gdf, y_coord, x_coord, marker_color='blue', marker_name='info', 
                  scale_minx=0, scale_maxx=23, datetime_col='datetime'):
    """Creates a ipyleaflet marker with a popup within. This popup is filled with a heatmap
    based on datetime values.
    
    Parameters
    ----------
    events_gdf : GeoDataFrame
        A GeoDataFrame with the geolocated events and their datetimes
    y_coord : str
        Y coordinate to place the marker
    x_coord : str
        X coordinate to place the marker
    marker_color : str
        The color of the marker
    marker_name : str
        The to give to the marker layer
    scale_minx : int
        Initial time interval
    scale_maxx : int
        Ending time interval
    datetime_col : str
        Datetime column to look at on the dataset
    """
    scale_dif = scale_maxx - scale_minx
    
    tmpfile = BytesIO()
    distribution_chart = matplot_datetime_heatmap(events_gdf, scale_minx=scale_minx, 
                                                  scale_maxx=scale_maxx, datetime_col=datetime_col)
    
    distribution_chart.savefig(tmpfile, format='png', bbox_inches='tight',dpi=100)
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    popup_width = max(200,scale_dif*20) 
    chart_html = '<img src=\'data:image/png;base64,{}\' width="{}" height="500"><br>'.format(
        encoded, popup_width)

    message = widgets.HTML()
    message.value = chart_html + \
                    'Number of crimes: '+str(len(events_gdf))


    marker = ipyleaflet.Marker(location = [y_coord, x_coord], 
                               popup=message, 
                               popup_min_width=popup_width,
                               icon=ipyleaflet.AwesomeIcon(marker_color=marker_color, name='info'),
                               color=marker_color, draggable=False)

    return marker

def route_statistic_popup(graph, total_events, events=None, heatmap=False, scale_minx=0, 
                          scale_maxx=23, datetime_col='datetime', round_trip_length=False):
    """Creates a marker with a popup, to be displayed on mean latitude and longitude calculated
    based on a graph. This popup shows the percentage of crimes inside a graph e the total 
    length of the graph.
    
    Attributes
    ----------
    graph: networkx graph
        The graph with number of crimes
    total_events : GeoDataFrame
        A GeoPandasDataframe containing all the events, used to compare with the smaller graph area
    events: GeoDataFrame
        Geopandas geodaframe with segments geometry and graph u and v nodes.
    heatmap : bool
        Boolean that defines if the popup should have a heatmao within or not
    scale_minx : int
        Initial time interval
    scale_maxx : int
        Ending time interval
    datetime_col : str
        Datetime column to look at on the dataset
    
    Returns
    -------
    A marker popup layer
    """
    
    x_mean = np.mean([x for _, x in graph.nodes(data='x')])
    y_mean = np.mean([y for _, y in graph.nodes(data='y')])

    
    number_of_crimes = int(graph.size(weight='score'))
    perc_crimes = str(round(number_of_crimes/len(total_events) * 100, 2))+'%'
    
    if round_trip_length:
        hotspot_length = round(round_trip_size(graph, weight='length'),1)
    else:
        hotspot_length = round(graph.size(weight='length'),1)
    
    message = widgets.HTML()
    message = widgets.HTML()
    message.value = '<br>Percentage of crimes: '+ perc_crimes + \
                    '<br>Route length: '+str(hotspot_length)+' m'
    
    if heatmap:
        marker_popup = heatmap_popup(events, y_mean, x_mean, marker_color='red', 
                                     scale_minx=scale_minx, scale_maxx=scale_maxx, 
                                     datetime_col=datetime_col)
        
        marker_popup.popup.value = marker_popup.popup.value + message.value
    else:
        wid_number_of_crimes = widgets.HTML()
        wid_number_of_crimes.value = 'Number of crimes: '+str(number_of_crimes)
        
        marker_popup = ipyleaflet.Marker(location = [y_mean, x_mean], 
                                   popup=message, 
                                   popup_min_width=0,
                                   icon=ipyleaflet.AwesomeIcon(marker_color='red', name='info'),
                                   color='red', draggable=False)
        
        marker_popup.popup.value = wid_number_of_crimes.value + marker_popup.popup.value

    return marker_popup

def ipyleaflet_fit_bounds(leaflet_map, geometry):
    """Change the center of the map based on shapely geometry
    
    Attributes
    ----------
    leaflet_map: ipyleaflet.leaflet.Map
        An instance of ipyleaflet map
    geometry: shapely Polygon
        Polygon used to center the map
    """
    centroid = geometry.exterior.centroid.xy
    new_center = [*centroid[1].tolist(), *centroid[0].tolist()]
    
    leaflet_map.center = new_center
    
def ipyleaflet_geometry_bounds(geometry, name='Geometry', stroke=True, color='black', weight=1, 
                               fill=True, fill_color=None, fill_opacity=0, line_cap='round', 
                               line_join='round'):
    """Create a ipyleaflet layer that shows the shapely geometry on top of the map.

    Attributes
    ----------
    name: str, default 'Geometry'
        The name of the layer.
    stroke: boolean, default True
        Whether to draw a stroke.
    color: CSS color, default '#0033FF'
        CSS color.
    weight: int, default 5
        Weight of the stroke.
    fill: boolean, default True
        Whether to fill the path with a flat color.
    fill_color: CSS color, default None
        Color used for filling the path shape. If None, the color attribute
        value is used.
    fill_opacity: float, default 0.2
        Opacity used for filling the path shape.
    line_cap: string, default "round"
        A string that defines shape to be used at the end of the stroke.
        Possible values are 'round', 'butt' or 'square'.
    line_join: string, default "round"
        A string that defines shape to be used at the corners of the stroke.
        Possible values are 'arcs', 'bevel', 'miter', 'miter-clip' or 'round'.
    """

    exterior = geometry.exterior.xy
    exterior_coordinates = [[exterior[1][i], exterior[0][i]] for i in range(len(exterior[0]))]
    border_layer = ipyleaflet.Polyline(locations=exterior_coordinates, 
                                       stroke=stroke,
                                       fill=fill,
                                       color=color,
                                       fill_color=fill_color,
                                       fill_opacity=fill_opacity,
                                       line_cap=line_cap,
                                       line_join=line_join,
                                       weight=weight)
    
    border_layer = ipyleaflet.LayerGroup(name=name, layers=[border_layer] )
    
    return border_layer

def ipyleaflet_map(tiles='cartodbpositron', layer_control=True, full_screen=True, 
                   scroll_wheel_zoom=True, zoom_control=True, basemap_https=False):
    """Create a simple ipyleaflet map
    
    Attributes
    ----------
    tiles: ipyleaflet tile layer
        Basemap used on the map
    layer_control: bool
        If a map should be created with a Layer Control or not
    full_screen: bool
        If a map should be created with a full screen Control or not
    scroll_wheel_zoom: bool
        Enable mouse wheel zoom
    zoom_control: bool
        Enable zoom control
    basemap_https: bool
        Force basemap request to use security layer with https
    """

    if tiles == 'cartodbpositron':
        basemap = ipyleaflet.basemaps.CartoDB.Positron
    else:
        raise TypeError("The map tiles {} was not implemented yet".format(tiles))

    # force basemap url to use security layer
    if basemap_https:
        text_match = re.search('(.+?)://', basemap['url'])
        if text_match and (text_match.group(1) == 'http'):
            basemap['url'] = basemap['url'].replace('http://', 'https://')

    lmap = ipyleaflet.Map(basemap=basemap, scroll_wheel_zoom=scroll_wheel_zoom, 
                          zoom_control=zoom_control)
    
    if full_screen:
        lmap.add_control(ipyleaflet.FullScreenControl())
    if layer_control:
        lmap.add_control(ipyleaflet.LayersControl(position='topright'))

    return lmap

def ipyleaflet_map_geometry(geometry, borders_name='Borders', tiles='cartodbpositron',
                            layer_control=True, full_screen=True, 
                            scroll_wheel_zoom=True, basemap_https=False):
    """Create an ipyleaflet map, with a shapely geometry border on top of it and centered based
    on that geometry.
    
    Attributes
    ----------
    geometry: shapely Polygon
        The polygon to be drawned and centered on top of the map
    borders_name: str
        The name of the geoemtry layer
    tiles: ipyleaflet tile layer
        Basemap used on the map
    layer_control: bool
        If a map should be created with a Layer Control or not
    full_screen: bool
        If a map should be created with a full screen Control or not
    scroll_wheel_zoom: bool
        Enable mouse wheel zoom
    basemap_https: bool
        Force basemap request to use security layer with https
    """
    
    # Create map
    lmap = ipyleaflet_map(tiles=tiles, layer_control=layer_control, full_screen=full_screen,
                          scroll_wheel_zoom=scroll_wheel_zoom, basemap_https=basemap_https)

    # Create the limit border
    borders = ipyleaflet_geometry_bounds(geometry, borders_name)
    lmap.add_layer(borders)
    # Fit the map zoom
    ipyleaflet_fit_bounds(lmap, geometry)
    
    return lmap

def ipyleaflet_clusterize_hot_segments(select_roads, events, hot_segments_name='Hot Segments', 
                                       clusters_name='Events clusters', maximum_cluster=100,
                                       event_popup_column=None):
    """Creates a dict with ipyleaflet Layers, these layers are built using two specialized
    functions that handles the creation of hot segments and creation of cluster markers.
    This method is a bulk create of layers, based on a list of graphs and events.
    
    Parameters
    ----------
    select_roads : list
        List of networkx graphs with the segment index within
    events : GeoDataFrame
        GeoDataFrame containing the events and the nearest edges index from it
    hot_segments_name : str
        Name of the hot_segments layer
    clusters_name : str
        Name of the clusters layer
    event_popup_column : str
        Event information column to be printed on the popup
    
    Returns
    -------
    layers = dict
        Returns a dict containing the layers created
    """
    
    layer_hot_segments = ipyleaflet.LayerGroup(name=hot_segments_name)
    layer_clusters = ipyleaflet.LayerGroup(name=clusters_name)    
    
    for rank, (index, hotspot_road) in enumerate(select_roads.iterrows()):
        filtered_events = events[events['nearest_edge'] == index]
        
        layer_hot_segments.add_layer(ipyleaflet_hotsegment(filtered_events, 
                                                           hotspot_road, 
                                                           rank, 
                                                           name=hot_segments_name))
        
        if rank < maximum_cluster:
            layer_clusters.add_layer(ipyleaflet_clusters(filtered_events, 
                                                         name=clusters_name,
                                                         popup_column=event_popup_column))
    
    layers = {}
    layers[hot_segments_name] = layer_hot_segments
    layers[clusters_name] = layer_clusters
    
    return layers


def ipyleaflet_hotsegment(gdf_events, road_geometry, segment_id, name='Hot Segments'):
    """Creates an ipyleaflet layer that represents a geolocated hot segment.
    
    Parameters
    ----------
    gdf_events : GeoDataFrame
        GeoDataframe with events founded near the hot segment
    road_geometry : GeoSerie
        GeoSerie contaning the geometry of the hot segment
    segment_id : int
        Segment index or indentifier, should be unique
    name : str
        Name to give to the layer
    
    Returns
    -------
    layer_hotsegment = ipyleaflet.Layer
        Returns the ipyleaflet Layer object, to be added on a map or in another layer 
    """
    layer_hotsegment = ipyleaflet.LayerGroup(name=name)
    
    message = widgets.HTML()
    message.value = 'Rank: '+str(segment_id+1)+'<br> Crimes: '+str(len(gdf_events))
    
    # Converto to GeoJson
    geojson_poly = gpd.GeoSeries(road_geometry['geometry']).to_json()
    line_location = json.loads(geojson_poly)['features'][0]['geometry']['coordinates']

    
    list(map(lambda a:a.reverse(), line_location))
    line = ipyleaflet.Polyline(locations = line_location, color='red', popup=message)
    layer_hotsegment.add_layer(line)
    
    return layer_hotsegment

def ipyleaflet_clusters(gdf_events, name='Event clusters', popup_column=None):
    """Creates an ipyleaflet layer that is a  cluster of markers built from events near from
     each other.
    
    Parameters
    ----------
    gdf_events : GeoDataFrame
        GeoDataframe with events founded near the hot segment
    name : str
        Name to give to the layer
    popup_column : str
        Event information column to be printed on the popup
    
    Returns
    -------
    layer_cluster = ipyleaflet.Layer
        Returns the ipyleaflet Layer object, to be added on a map or in another layer 
    """
    layer_cluster = ipyleaflet.LayerGroup(name=name, show=False)


    markers = [ipyleaflet.CircleMarker(
                        location=[event.geometry.y, event.geometry.x],
                        radius=2,
                        color='red',
                        fill=True,
                        fill_opacity=0.7) for _, event in gdf_events.iterrows()]
    
    if popup_column:
        for index, (_, event) in enumerate(gdf_events.iterrows()):
            markers[index].popup = widgets.HTML(str(event.loc[popup_column]))
        
    mc = ipyleaflet.MarkerCluster(markers=markers)
    layer_cluster.add_layer(mc)

    return layer_cluster


def ipyleaflet_routes_with_statistics(hotspots_list, gdf_events, graph, 
    hotspots_index=None, round_trip_length=False):
    """Given a list of graph routes, creates all layers needed to plot the route, route
    statistics and the statistics aggregated for all of those routes.
    
    Parameters
    ----------
    hotspot_list : list
        List with all graph routes and scores
    gdf_events : GeoDataFrame
        GeoDataFrame with all events found in those routes
    graph : networkx graph
        Networkx graph of the study area
    hotspot_index : int
        If if defined it will plot only the specific route found on this index
    """
    colors_map = ['red','blue','green','black','orange','cyan',
                  'purple','darkred','darkblue','darkgreen']
    
    layer_routes = ipyleaflet.LayerGroup(name='Patrol Routes')
    layer_statistics = ipyleaflet.LayerGroup(name='Routes Statistics')
    layer_agg_statistics = ipyleaflet.LayerGroup(name='Aggregate Statistics')


    all_edges = gpd.GeoDataFrame()
    
    for index, hotspot in enumerate(hotspots_list):
        if hotspots_index is not None:
            if index+1 not in hotspots_index:
                continue
        
        hotspot.add_nodes_from((n, graph.nodes[n]) for n in hotspot.nodes)
        hotspot.graph = graph.graph
        hotspot_nodes, hotspot_edges = graph_to_gdfs(hotspot)
        
        all_edges = all_edges.append(hotspot_edges)
        
        color = colors_map[index % len(colors_map)]
        
        #Create routes
        layer_routes.add_layer(ipyleaflet_patrol_routes(gdf_events, hotspot, nodes_color=color))
        
        #Create routes statistics
        layer_statistics.add_layer(ipyleaflet_patrol_route_statistics(hotspot, gdf_events, index, 
                                                                      round_trip_length=round_trip_length))
    
    #Create the aggregated statistics for the routes
    layer_agg_statistics.add_layer(ipyleaflet_agreggated_statistics(all_edges, gdf_events))
    
    layers = {}
    layers['routes'] = layer_routes
    layers['statistics'] = layer_statistics
    layers['agg_statistics'] = layer_agg_statistics


    return layers

def ipyleaflet_patrol_routes(gdf_events, hotspot, nodes_color='red', edge_color='black', 
                         edge_weight=1, edge_opacity=0.0, node_radius=1, 
                         node_opacity=0.7, node_fill=True):
    """Creates an ipyleaflet layer with the plot of patrol routes inside of it.
    
    Parameters
    ----------
    gdf_events : GeoDataFrame
        GeoDataFrame containing the events inside the route
    hotspot : networkx graph
        Networkx graph with the route segments, nodes and scores
    nodes_color : str
        Color to use on the nodes when plotting
    edge_color : str 
        Color to use on the edges when plotting
    edge_weight : int
        The thickness of the edge line
    edge_opacity : float
        Opacity to attribute to the edge segment
    node_radius : int
        The size of the node radius
    node_opacity : float
        Opacity to attribute to the node point
    node_fill : bool
        Defines if a node has to be a color filled or not
    """
    
    layer_routes = ipyleaflet.LayerGroup(name='Patrol Routes')
    
    hotspot_nodes, hotspot_edges = graph_to_gdfs(hotspot)

    hotspot_plot = ipyleaflet.GeoJSON(data=json.loads(hotspot_edges.to_json()),
                                      style={
                                          'color': edge_color,
                                          'weight': edge_weight,
                                          'fillOpacity': edge_opacity
                                      })
    
    layer_routes.add_layer(hotspot_plot)

    for node in hotspot_nodes['geometry']:
        marker = ipyleaflet.CircleMarker(
            location=[node.y, node.x],
            radius=node_radius,
            color=nodes_color,
            fill=node_fill,
            fill_opacity=node_opacity
            )
        layer_routes.add_layer(marker)
    
    return layer_routes

def ipyleaflet_patrol_route_statistics(hotspot, gdf_events, route_id, round_trip_length=False):
    """Creates a popup with route statistics like, length, number of crimes and percentage of crimes
    
    Parameters
    ----------
    hotspot : networkx graph
        Graph contaning the route edges, nodes and scores
    gdf_events : GeoDataFrame
        GeoDataFrame with all the events, used to compare with the events founded on route
    route_id : int
        An id to give to the route and use on the popup
    """
    layer_statistics = ipyleaflet.LayerGroup(name='Routes Statistics')

    total_of_events = len(gdf_events)
    hotspot_nodes, hotspot_edges = graph_to_gdfs(hotspot)

    number_of_crimes = int(sum(hotspot_edges['score']))
    perc_crimes = str(round(number_of_crimes/total_of_events * 100, 2))+'%'
    
    if round_trip_length:
        hotspot_length = round(round_trip_size(hotspot, weight='length'),1)
    else:
        hotspot_length = round(route_graph.size(weight='length'),1)

    message = widgets.HTML()
    message.value = 'Patrol Route '+str(route_id+1)+ \
                    '<br>Crimes: '+str(number_of_crimes)+'<br>Percentage: '+ \
                    perc_crimes+'<br>Route Length: '+str(hotspot_length)+' m'

    marker = ipyleaflet.Marker(location = [hotspot_edges.centroid.y.mean(), 
                               hotspot_edges.centroid.x.mean()],popup=message, popup_max_width=130,
                               icon = ipyleaflet.AwesomeIcon(marker_color='red', name='info'),
                               color='red', draggable=False)

    layer_statistics.add_layer(marker)
    
    return layer_statistics

def ipyleaflet_agreggated_statistics(all_edges, gdf_events):
    """Creates a convex hull of all aggregated routes and plot the statistic for those.
    
    Parameters
    ----------
    all_edges : GeoDataFrame
        All edges segments within those routes and their scores
    gdf_events : GeoDataFrame
        All events fouded on those routes.
    """
    layer_agg_statistics = ipyleaflet.LayerGroup(name='Aggregate Statistics')
    
    total_of_events = len(gdf_events)
    
    number_of_crimes = int(sum(all_edges['score']))
    perc_crimes = str(round(number_of_crimes/total_of_events * 100, 2))+'%'
    hotspot_length = round(sum(all_edges['length']),1)
    
    message = widgets.HTML()
    message.value = 'Aggregate Statistics<br>Crimes: '+str(number_of_crimes)+'<br>Percentage: '+ perc_crimes+ \
                    '<br>Length: '+str(hotspot_length)+' m'
    
    convex_geometry = all_edges.unary_union.convex_hull
    exterior = convex_geometry.exterior.xy

    convex_polygon = ipyleaflet.Polygon(locations= [[exterior[1][i], exterior[0][i]] 
                                                     for i in range(len(exterior[0]))],
                                        stroke=False,
                                        fill_opacity=0.2,
                                        fill_color='blue',
                                        popup=message)
    
    layer_agg_statistics.add_layer(convex_polygon)
    
    return layer_agg_statistics

def ipyleaflet_plot_statistic(graph_routes, gdf_events, layer_name='Routes Statistics', 
                              update_layer=None, initial_hour_wid=None, end_hour_wid=None):
    """Creates an ipyleaflet layer with route statistics and heatmap
    
    Parameters
    ----------
    graph_routes : networkx graph
        Graph that represents the route
    gdf_events : GeoDataFrame
        All events founded on the routes
    layer_name : str
        Layer name to give to the ipyleaflet layer
    update_layer : bool
        Defines if the function should update an already created layer or create a new one
    """
    
    layer_statistics = ipyleaflet.LayerGroup(name=layer_name)

    total_of_events = len(gdf_events)
    
    if isinstance(graph_routes, dict):
        routes_list = graph_routes.values()
    elif isinstance(graph_routes, list):
        routes_list = graph_routes

    for route_graph in routes_list:

        edges_indexes = [edge_index for _, _, edge_index in route_graph.edges(data='index')]

        events_tmp = gdf_events[gdf_events['nearest_edge'].isin(edges_indexes)]
        print(events_tmp.empty)
        
        if not events_tmp.empty:      
            scale_minx = initial_hour_wid.value.split(':')[0] if initial_hour_wid is not None else 0
            scale_maxx = end_hour_wid.value.split(':')[0] if end_hour_wid is not None else 23
            
            marker = route_statistic_popup(route_graph, gdf_events, events_tmp, heatmap=True, 
                                           scale_minx=scale_minx, scale_maxx=scale_maxx)    
        else:
            marker = route_statistic_popup(route_graph,gdf_events)  

        layer_statistics.add_layer(marker)
        
    if update_layer is not None:
        update_layer.layers = layer_statistics.layers
    else:    
        return layer_statistics


def create_statistics_marker(route_graph, gdf_events, initial_hour, 
                            final_hour, layer_name='Routes Statistics', 
                            round_trip_length=False):

    total_of_events = len(gdf_events)

    edges_indexes = [edge_index for _, _, edge_index in route_graph.edges(data='index')]

    events_tmp = gdf_events[gdf_events['nearest_edge'].isin(edges_indexes)]

    if not events_tmp.empty:

        scale_minx = int(initial_hour.split(':')[0])
        scale_maxx = int(final_hour.split(':')[0])

        scale_dif = scale_maxx - scale_minx

        tmpfile = BytesIO()
        distribution_chart = matplot_datetime_heatmap(events_tmp, scale_minx=scale_minx, scale_maxx=scale_maxx)
        distribution_chart.savefig(tmpfile, format='png', bbox_inches='tight',dpi=100)
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')

        popup_width = max(200,scale_dif*20) 
        chart_html = '<img src=\'data:image/png;base64,{}\' width="{}" height="270"><br>'.format(encoded, popup_width)

    else:
        popup_width = 0
        chart_html = ''

    number_of_crimes = int(route_graph.size(weight='score'))
    perc_crimes = str(round(number_of_crimes/total_of_events * 100, 2))+'%'

    if round_trip_length:
        hotspot_length = round(round_trip_size(route_graph, weight='length'),1)
    else:
        hotspot_length = round(route_graph.size(weight='length'),1)

    message = widgets.HTML()
    message.value = chart_html + \
                    'Number of crimes: '+str(number_of_crimes) + \
                    '<br>Percentage of crimes: '+ perc_crimes + \
                    '<br>Route length: '+str(hotspot_length)+' m'

    x_mean = np.mean([x for _, x in route_graph.nodes(data='x')])
    y_mean = np.mean([y for _, y in route_graph.nodes(data='y')])

    marker = ipyleaflet.Marker(location = [y_mean, x_mean], 
                                popup=message, 
                                popup_min_width=popup_width,
                                icon=ipyleaflet.AwesomeIcon(marker_color='red', name='info'),
                                color='red', draggable=False)

    return marker

def create_statistics_polygon(graphs_dict, group_indexes, gdf_events, initial_hour, final_hour, 
                            layer_name='Group Statistics', update_layer=None):

    total_of_events = len(gdf_events)

    edges_indexes = [edge_index for route_index in group_indexes
                        for u, v, edge_index in graphs_dict[route_index].edges(data='index')]

    events_tmp = gdf_events[gdf_events['nearest_edge'].isin(edges_indexes)]

    if not events_tmp.empty:

        scale_minx = int(initial_hour.split(':')[0])
        scale_maxx = int(final_hour.split(':')[0])

        scale_dif = scale_maxx - scale_minx

        tmpfile = BytesIO()
        distribution_chart = matplot_datetime_heatmap(events_tmp, scale_minx=scale_minx, scale_maxx=scale_maxx)
        distribution_chart.savefig(tmpfile, format='png', bbox_inches='tight',dpi=100)
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')

        popup_width = max(200,scale_dif*20) 
        chart_html = '<img src=\'data:image/png;base64,{}\' width="{}" height="270"><br>'.format(encoded, popup_width)

    else:
        popup_width = 0
        chart_html = ''

    number_of_crimes = int(sum([graphs_dict[route_index].size(weight='score') 
                                for route_index in group_indexes]))
    perc_crimes = str(round(number_of_crimes/total_of_events * 100, 2))+'%'
    hotspot_length = round(sum([graphs_dict[route_index].size(weight='length') 
                                for route_index in group_indexes]),1)

    message = widgets.HTML()
    message.value = chart_html + \
                    'Number of crimes: '+str(number_of_crimes) + \
                    '<br>Percentage of crimes: '+ perc_crimes + \
                    '<br>Sum of routes length: '+str(hotspot_length)+' m'

    x_coords = [x for route_index in group_indexes 
                for node, x in graphs_dict[route_index].nodes(data='x')]

    y_coords = [y for route_index in group_indexes 
                for node, y in graphs_dict[route_index].nodes(data='y')]

    #x_mean = np.mean(x_coords)
    #y_mean = np.mean(y_coords)

    mpt = MultiPoint([Point(x,y) for x, y in zip(x_coords, y_coords)])
    convex_geometry = mpt.convex_hull

    mpt = MultiPoint([Point(x,y) for x, y in zip(x_coords, y_coords)])
    convex_geometry = mpt.convex_hull
    convex_x, convex_y = convex_geometry.exterior.xy
    convex_polygon = ipyleaflet.Polygon(locations= [[y, x] for y, x in zip(convex_y, convex_x)], 
                                        color="green",
                                        fill_color="green",
                                        weight=2,
                                        fill_opacity=0.2,
                                        popup=message,
                                        popup_min_width=popup_width)

    return convex_polygon