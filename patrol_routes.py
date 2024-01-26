import networkx as nx
from copy import deepcopy
from itertools import chain
from sys import float_info
import osmnx as ox
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def chinese_postman_circuit(graph, weight=None, source=None):
    
    """
    Returns a shortest closed circuit that visits every edge of a connected undirected graph.
    This problema is called Chinese postman problem, postman tour or route inspection problem.
    
    Parameters
    ----------
    graph : networkx.Graph
        Networkx graph.
    weight : string or None, (default=None)
        The edge attribute that holds the numerical value used as a weight. 
        If None, then each edge has weight 1.
    source : node, optional
        Starting node for circuit.
    
    Returns
    -------
    postman_circuit : list
        A sequence of edges that form the postman's circuit.
        
    Notes
    -----
    This solution was proposed in the route inspection problem on 
    https://en.wikipedia.org/wiki/Route_inspection_problem
    https://www.datacamp.com/community/tutorials/networkx-python-graph-tutorial
    """
    
    if nx.is_directed(graph):
        raise Exception("Expected undirected networkx graph, but {} was found".format(type(graph)))

    # Calculate list of nodes with odd degree
    nodes_odd_degree = [node for node, degree in nx.degree(graph) if degree % 2 == 1]
    
    if not nodes_odd_degree:
        postman_circuit = []
        eulerian_circuit = nx.eulerian_circuit(graph, source=source)
        for edge in eulerian_circuit:
            edge_att = graph[edge[0]][edge[1]]
            postman_circuit.append((edge[0], edge[1], edge_att))
        return postman_circuit
    else:
        # Compute all pairs of odd nodes. in a list of tuples
        odd_node_pairs = list(itertools.combinations(nodes_odd_degree, 2))
        # Compute shortest distance between each pair of nodes in a graph.  
        distances_dict = {}
        for pair in odd_node_pairs:
            distances_dict[pair] = nx.dijkstra_path_length(graph, pair[0], pair[1], weight=weight)
            
        # Create a completely connected graph using a list of vertex pairs 
        # and the shortest path distances between them
        complete_graph = nx.Graph()
        for edge, distance in distances_dict.items():
            # flip weights
            wt_i = - distance
            complete_graph.add_edge(edge[0], edge[1], distance=distance, weight=wt_i)
        # Compute min weight matching.
        odd_matching = nx.algorithms.max_weight_matching(complete_graph, True)
        
        # Add the min weight matching edges to the original graph
        # We need to make the augmented graph a MultiGraph so we can add parallel edges
        graph_aug = nx.MultiGraph(graph.copy())
        for pair in odd_matching:
            distance_pair = distances_dict[(pair[0], pair[1])] if (pair[0], pair[1]) in distances_dict else distances_dict[(pair[1], pair[0])]
            graph_aug.add_edge(pair[0], pair[1], distance=distance_pair, trail_aux='augmented')
        
        # Create the eulerian path using only edges from the original graph.
        postman_circuit = []
        
        eulerian_circuit = nx.eulerian_circuit(graph_aug, source=source)
        for edge in eulerian_circuit:
            edge_data = graph_aug.get_edge_data(edge[0], edge[1])    
            
            if ('trail_aux' not in edge_data[0]):
                # If `edge` exists in original graph, grab the edge attributes and add to postman circuit.
                edge_att = graph[edge[0]][edge[1]]
                postman_circuit.append((edge[0], edge[1], edge_att))
            else:
                aug_path = nx.shortest_path(graph, edge[0], edge[1], weight='distance')
                aug_path_pairs = list(zip(aug_path[:-1], aug_path[1:]))

                # If `edge` does not exist in original graph, find the shortest path between its nodes and
                # add the edge attributes for each link in the shortest path.
                for edge_aug in aug_path_pairs:
                    edge_aug_att = graph[edge_aug[0]][edge_aug[1]]
                    postman_circuit.append((edge_aug[0], edge_aug[1], edge_aug_att))
    
    return postman_circuit

def intersection_minimum_cycle_basis(graph):
    
    if not isinstance(graph, nx.Graph):
        raise Exception("Expected nx.Graph, but {} was found".format(type(graph)))

    # get the minimum cycle basis
    cycle_basis = nx.minimum_cycle_basis(graph)
    
    # minimum cycle basis do not return nodes in order, 
    # induce the subgraph and take the edges
    cycle_basis_edges = [list(nx.induced_subgraph(graph, cycle).edges)
                         for cycle in cycle_basis]

    intersection_edges = []
    for index, cycle_edges in enumerate(cycle_basis_edges):

        # get other cycles of the graph to compare
        other_cycles_edges = cycle_basis_edges[index+1:]
        # interate over the other cycles
        for check_cycle in other_cycles_edges:
            for edge in cycle_edges:
                if (edge in check_cycle) or (edge[::-1] in check_cycle):
                    # not add repeat edges
                    if (edge not in intersection_edges) and (edge[::-1] not in intersection_edges):
                        intersection_edges.append(edge)

    return intersection_edges

def percentage_route_cycle(graph, weight='length'):
    
    if not isinstance(graph, nx.Graph):
        raise Exception("Expected nx.Graph, but {} was found".format(type(graph)))
    
    # get the minimum cycle basis
    cycle_basis = nx.minimum_cycle_basis(graph)
    cycle_size = sum([nx.induced_subgraph(graph, cycle).size(weight=weight)
                      for cycle in cycle_basis])
    # cycles count once
    intersection_edges = intersection_minimum_cycle_basis(graph)
    cycle_size -= sum([graph[edge[0]][edge[1]][weight] for edge in intersection_edges])
    
    route_size = graph.size(weight=weight)
    perc_of_cycle = 100*(cycle_size/route_size)
    
    return perc_of_cycle

def route_cycle_size(graph, weight='length'):
    
    if not isinstance(graph, nx.Graph):
        raise Exception("Expected nx.Graph, but {} was found".format(type(graph)))
    
    # get the minimum cycle basis
    cycle_basis = nx.minimum_cycle_basis(graph)
    cycle_size = sum([nx.induced_subgraph(graph, cycle).size(weight=weight)
                      for cycle in cycle_basis])
    # cycles count once
    intersection_edges = intersection_minimum_cycle_basis(graph)
    cycle_size -= sum([graph[edge[0]][edge[1]][weight] for edge in intersection_edges])
    
    return cycle_size

def create_root_subgraph(graph, edge, max_length):
    u, v = edge
    root_length = graph.edges[edge]['length']
    u_dists, _ = nx.single_source_dijkstra(graph, u, cutoff=max_length - root_length, weight='length')
    v_dists, _ = nx.single_source_dijkstra(graph, v, cutoff=max_length - root_length, weight='length')
    subgraph_nodes = {u, v}
    for w in u_dists:
        if w in v_dists:
            if u_dists[w] + v_dists[w] + root_length < max_length + float_info.epsilon:
                subgraph_nodes.add(w)
    subgraph = graph.subgraph(subgraph_nodes)

    return subgraph

def add_edge_subgraph(graph, subgraph, edge):
    u, v = edge
    edge_data = graph[u][v]
    subgraph.add_edge(u, v, **edge_data)

def add_path_subgraph(graph, subgraph, add_path):
    if len(add_path) > 1:
        for prev_node, curr_node in zip(add_path, add_path[1:]):
            tmp_edge = (prev_node, curr_node)
            add_edge_subgraph(graph, subgraph, tmp_edge)
                    
def sum_weight_path(graph, path, weight):
    if len(path) > 1:
        path_size = sum(graph[u][v][weight] for u, v in zip(path, path[1:]))
    else:
        path_size = 0
        
    return path_size

def round_trip_size(graph, weight='length'):
    
    if not isinstance(graph, nx.Graph):
        raise Exception("Expected nx.Graph, but {} was found".format(type(graph)))
    
    postman_circuit = chinese_postman_circuit(graph, weight=weight)
    route_size = sum([data[weight] for u, v, data in postman_circuit])
    
    return route_size

# def round_trip_size(graph, weight='length'):
    
#     if not isinstance(graph, nx.Graph):
#         raise Exception("Expected nx.Graph, but {} was found".format(type(graph)))
    
#     # get the minimum cycle basis
#     cycle_basis = nx.minimum_cycle_basis(graph)
#     cycle_basis_size = sum([nx.induced_subgraph(graph, cycle).size(weight=weight)
#                             for cycle in cycle_basis])
    
#     # edge that does not belong to any cycle count twice
#     route_size = cycle_basis_size + 2*sum([graph[edge[0]][edge[1]][weight] 
#                                            for edge in nx.bridges(graph)])
    
#     return route_size

def patrol_routes_metrics(routes_list, graph, hotsegments=None, weight_length='length', 
                          weight_score=None, routes_length_metric=True, hotsegments_score=True, score_npai=True, 
                          cycle_coverege_length=True, round_trip_route=True, percentage=False):
    
    if not isinstance(graph, nx.Graph):
        raise Exception("Expected nx.Graph, but {} was found".format(type(graph)))
    
    # Initialize variables as zero to check reusability
    routes_length = 0
    routes_score = 0
    graph_score = 0
    
    df_metrics = pd.DataFrame(columns=['metrics'])
    
    if routes_length_metric:
        if round_trip_route:
            routes_length_vec = [round_trip_size(route) for route in routes_list]
        else:
            routes_length_vec = [route.size(weight=weight_length) for route in routes_list]
        
        df_metrics.loc['Mean of patrol routes length (std)'] = '{} ({:.2f})'.format(int(np.mean(routes_length_vec)), 
                                                                            np.std(routes_length_vec))
    
    # Calculate the sum of scores of the routes
    if weight_score is not None:
        routes_score = sum([route.size(weight=weight_score) for route in routes_list])
        
        if percentage:
            graph_score = graph.size(weight=weight_score)
            routes_score_percentage = 100*(routes_score/graph_score)
            df_metrics.loc['Patrol routes score (%)'] = '{} ({:.2f}%)'.format(int(routes_score), 
                                                                              routes_score_percentage)
        else:
            df_metrics.loc['Patrol routes score'] = routes_score
    
    # Calculate the proportion of routes length that is hotsegment
    if hotsegments is not None:
        
        routes_segments = [(u, v) for route in routes_list
                           for u,v in route.edges()]
        
        hotsegments_covered = [segment for segment in routes_segments 
                               if (segment in hotsegments) or (segment[::-1] in hotsegments)]
        
        num_hotsegments_covered = len(hotsegments_covered)
        
        
        hotsegments_covered_length = sum([graph[u][v][weight_length] for u,v in hotsegments_covered])
        
        if round_trip_route:
            # bridge segments or inside a cycle count twice for route length (round trip)
            hotsegments_covered_length += sum([graph[edge[0]][edge[1]][weight_length] 
                                              for route in routes_list 
                                              for edge in list(nx.bridges(route)) +
                                                          intersection_minimum_cycle_basis(route)
                                              if (edge in hotsegments_covered) or 
                                                 (edge[::-1] in hotsegments_covered)])
        
        if (weight_score is not None) and hotsegments_score:
            hotsegments_covered_score = sum([graph[u][v][weight_score] for u,v in hotsegments_covered])
        
        if percentage:
            if (weight_score is not None) and hotsegments_score:
                hotsegments_score = sum([graph[u][v][weight_score] for u,v in hotsegments])
                covered_score_percentage = 100*(hotsegments_covered_score/hotsegments_score)

                df_metrics.loc['Hotsegments covered score (%)'] = '{} ({:.2f}%)'.format(int(hotsegments_covered_score), 
                                                                                        covered_score_percentage)
            
            num_hotsegments_covered_percentage = 100*(num_hotsegments_covered/len(hotsegments))
            df_metrics.loc['Number hotsegments covered (%)'] = '{} ({:.2f}%)'.format(num_hotsegments_covered,
                                                                              num_hotsegments_covered_percentage)
            
            if round_trip_route:
                routes_length = sum([round_trip_size(route) for route in routes_list])
            else:
                routes_length = sum([route.size(weight=weight_length) for route in routes_list])
               
            hotsegments_covered_percentage = 100*(hotsegments_covered_length/routes_length)
            
            
            df_metrics.loc['Patrol length as hotsegment (%)'] = '{} ({:.2f}%)'.format(int(hotsegments_covered_length), 
                                                                                      hotsegments_covered_percentage) 
        else:
            if (weight_score is not None) and hotsegments_score:
                df_metrics.loc['Hotsegments covered score'] = hotsegments_covered_score
                
            df_metrics.loc['Number hotsegments covered'] = num_hotsegments_covered
            df_metrics.loc['Patrol length as hotsegment'] = hotsegments_covered_length
                     
            
    # Calculate the proportion of routes that are included in a cycle
    if cycle_coverege_length:
        
        routes_cycle_length = sum([route_cycle_size(route, weight_length) for route in routes_list])
        routes_length = sum([route.size(weight=weight_length) for route in routes_list])
            
        if percentage:
            routes_cycle_percentage = 100*(routes_cycle_length/routes_length)
            df_metrics.loc['Patrol cycle length (%)'] = '{} ({:.2f}%)'.format(int(routes_cycle_length), 
                                                                              routes_cycle_percentage)
        else:
            df_metrics.loc['Patrol cycle length'] = routes_cycle_length
    
    # Calculate overall network predictive accuracy index
    if score_npai:
        if not routes_score:
            routes_score = sum([route.size(weight=weight_score) for route in routes_list])
        if not graph_score:
            graph_score = graph.size(weight=weight_score)
        if not routes_length:    
            routes_length = sum([route.size(weight=weight_length) for route in routes_list])
            
        graph_length = graph.size(weight=weight_length)

        npai = (routes_score / graph_score) / (routes_length / graph_length)
            
        df_metrics.loc['Network PAI'] = '{:.2f}'.format(npai)    
    
    return df_metrics


def min_hotsegment_cycle_path(root_subgraph, patrol_subgraph, root_hotsegments, neighb_hotsegments, 
                              available_distance, debug=False):
    
    if debug:
        print('Neighb_hotsegments: ', neighb_hotsegments)

    # create auxiliar hotnodes list 
    neighb_hotnodes = list(chain(*neighb_hotsegments))

    # calculate all paths from patrol to the other nodes
    all_dijk_path = nx.multi_source_dijkstra(root_subgraph, sources=list(patrol_subgraph.nodes), 
                                             cutoff=available_distance, weight='length')

    path_candidates_length = {node:all_dijk_path[0][node] 
                              for node in neighb_hotnodes 
                              if node in all_dijk_path[0].keys()}

    min_length = available_distance
    selected_path = {}

    for check_hotnode in path_candidates_length.keys():

        path_length = path_candidates_length[check_hotnode]

        if path_length > min_length:
            continue

        # get the hotnode adjacent to the checked node
        index_hotnode = neighb_hotnodes.index(check_hotnode)
        if (index_hotnode % 2) == 0:
            adj_hotnode = neighb_hotnodes[index_hotnode+1]
        else:
            adj_hotnode = neighb_hotnodes[index_hotnode-1]

        path_length += root_subgraph[check_hotnode][adj_hotnode]['length']

        if path_length < min_length:

            path_from_patrol = all_dijk_path[1][check_hotnode]
            if adj_hotnode in path_from_patrol:
                continue
                
            path_from_patrol += [adj_hotnode]
            
            # Create auxiliar graph to take the calculate the return simple path
            root_subgraph_aux = deepcopy(root_subgraph)
            # Remove the path_from_patrol from the simple path search graph 
            
            edges_ebunch = [(u,v) for u, v in zip(path_from_patrol, path_from_patrol[1:])]
            root_subgraph_aux.remove_edges_from(edges_ebunch)
            
            back_dist = available_distance
            back_path = []
            
            # if patrol subgraph is formed by one edge, give preference to go back to them
            if patrol_subgraph.number_of_edges() == 1:
                # give the preference to get a route from the endpoint to another
                priority_nodes = [node for node in list(patrol_subgraph.nodes) 
                                  if node not in [adj_hotnode, check_hotnode, path_from_patrol[0]]]

                for priority_node in priority_nodes:
                    if debug:
                        print("Priority node:", priority_node)
                    try:
                        pr_dist, pr_path = nx.single_source_dijkstra(root_subgraph_aux, 
                                                                     source=adj_hotnode,
                                                                     target=priority_node,
                                                                     cutoff=back_dist, 
                                                                     weight='length')
                        if pr_dist < back_dist:
                            back_dist = pr_dist
                            back_path = pr_path

                    except nx.NetworkXNoPath as e:
                        pass
            
            if not back_path:
                try:
                    back_dist, back_path = nx.multi_source_dijkstra(root_subgraph_aux, 
                                                                    sources=list(patrol_subgraph.nodes),
                                                                    target=adj_hotnode,
                                                                    cutoff=available_distance, 
                                                                    weight='length')
                    back_path = back_path[::-1]
                except nx.NetworkXNoPath as e:
                    pass
                
            if back_path and (len(back_path) > 1):
                
                if debug:
                    print("[New select route]")
                    print("Path from patrol:", path_from_patrol)
                    print("Path backing to patrol", back_path)
                
                # add the path to patrol ------------------------------------
                tmp_patrol_subgraph = deepcopy(patrol_subgraph)

                before_patrol_length = round_trip_size(tmp_patrol_subgraph)
                add_path = path_from_patrol + back_path[1:]
                add_path_subgraph(root_subgraph, tmp_patrol_subgraph, add_path)

                # remove the intersection between basis cycles if not hotsegment
                intersection_cycle_edges = intersection_minimum_cycle_basis(tmp_patrol_subgraph)
                for edge in intersection_cycle_edges:
                    if (edge not in root_hotsegments) and (edge[::-1] not in root_hotsegments):
                        check_patrol_subgraph = deepcopy(tmp_patrol_subgraph)
                        check_patrol_subgraph.remove_edge(edge[0], edge[1])
                        # check if removing the edge the grap is disconnected
                        if nx.is_connected(check_patrol_subgraph):
                            tmp_patrol_subgraph = check_patrol_subgraph
                            # remove isolated nodes
                            tmp_patrol_subgraph.remove_nodes_from(list(nx.isolates(tmp_patrol_subgraph)))

                patrol_route_length = round_trip_size(tmp_patrol_subgraph)
                route_length = patrol_route_length - before_patrol_length
                
                if route_length < min_length:
                    min_length = route_length

                    selected_path = {'add_path': add_path, 'patrol_subgraph': tmp_patrol_subgraph,
                                     'patrol_route_length': patrol_route_length}

                   
    return selected_path


def min_hotsegment_path(graph, patrol_subgraph, hotsegments, 
                        available_distance, debug=False):

    # create auxiliar hotnodes list 
    neighb_hotnodes = list(chain(*hotsegments))

    # divide by 2 to consider round trip
    min_length = available_distance / 2

    # calculate all paths from patrol to the other nodes
    all_dijk_path = nx.multi_source_dijkstra(graph, sources=list(patrol_subgraph.nodes), 
                                             cutoff=min_length, weight='length')

    path_candidates_length = {node:all_dijk_path[0][node] 
                              for node in neighb_hotnodes 
                              if node in all_dijk_path[0].keys()}
    
    selected_path = {}

    for check_hotnode in path_candidates_length.keys():
        
        # get the hotnode adjacent to the checked node
        index_hotnode = neighb_hotnodes.index(check_hotnode)
        if (index_hotnode % 2) == 0:
            adj_hotnode = neighb_hotnodes[index_hotnode+1]
        else:
            adj_hotnode = neighb_hotnodes[index_hotnode-1]

        path = all_dijk_path[1][check_hotnode]
        if adj_hotnode in path:
            continue
            
        path_length = path_candidates_length[check_hotnode]
        path_length += graph[check_hotnode][adj_hotnode]['length']

        if path_length < min_length:
            
            path += [adj_hotnode]
            min_length = path_length

            # add the path to patrol ------------------------------------
            tmp_patrol_subgraph = deepcopy(patrol_subgraph)
            add_path_subgraph(graph, tmp_patrol_subgraph, path)

            patrol_route_length = round_trip_size(tmp_patrol_subgraph)

            selected_path = {'add_path': path, 'patrol_subgraph': tmp_patrol_subgraph,
                             'patrol_route_length': patrol_route_length}
    
    return selected_path

class PatrolRoutes():
    
    def __init__(self, graph, weight='length', debug=False):
        
        self.graph = graph
        self.weight = weight
        self.debug = debug
        self.hotsegments = None
        self.excluded_seed = None
    
    def _exclude_hotsegments(self, patrol_subgraph):
        patrol_routes_edges = [(u,v) for u,v in patrol_subgraph.edges()]
        # update hotsegments
        self.hotsegments = [(u,v) for u, v in self.hotsegments 
                            if ((u,v) not in patrol_routes_edges) and
                               ((v,u) not in patrol_routes_edges)]
        # update excluded seed hotsegments
        self.excluded_seed = [(u,v) for u, v in self.excluded_seed
                              if ((u,v) not in patrol_routes_edges) and
                                 ((v,u) not in patrol_routes_edges)]
        
    def _delete_route_candidates(self, delete_indexes):
        self.all_routes_subgraph = np.delete(self.all_routes_subgraph, delete_indexes)
        self.all_routes_score = np.delete(self.all_routes_score, delete_indexes)
        self.all_seed_hotsegment = np.delete(self.all_seed_hotsegment, delete_indexes)
        
    def create_routes(self, num_routes, hotsegments, min_length, max_length, direct_path=True):
        
        self.hotsegments = [e for e in hotsegments if e[0] != e[1]]
        
        self.min_length = min_length
        self.max_length = max_length
        
        self.patrol_routes_list = []
        self.excluded_seed = []
        
        while len(self.patrol_routes_list) < num_routes:
            
            if len(self.hotsegments) > 0:
                seed_hotsegment = self.hotsegments[0]
                self.hotsegments.remove(seed_hotsegment)
            else:
                break
            
            if self.debug: 
                print("-- Seed: {}, Created routes: {}".format(seed_hotsegment,
                                                               len(self.patrol_routes_list)))
            
            # add excluded seed hotsegments to check neighboring
            check_hotsegments = self.hotsegments + self.excluded_seed
            patrol_subgraph = self.build_route(seed_hotsegment, check_hotsegments, 
                                               min_length, max_length)
            if patrol_subgraph is not False: 
                self.patrol_routes_list.append(patrol_subgraph)
                self._exclude_hotsegments(patrol_subgraph)
                self.graph.remove_edges_from(patrol_subgraph.edges)
            else:
                self.excluded_seed.append(seed_hotsegment)
        
        if direct_path:
            self._add_direct_path()

        return self

    def create_best_routes(self, num_routes, hotsegments, min_length, max_length, 
                      direct_path=True):
        
        self.hotsegments = [e for e in hotsegments if e[0] != e[1]]
        
        self.min_length = min_length
        self.max_length = max_length
        
        self.patrol_routes_list = []
        self.excluded_seed = []
        
        # define vectors to store the routes
        self.all_routes_score = np.zeros(len(self.hotsegments))
        self.all_routes_subgraph = np.empty(len(self.hotsegments), dtype=object)
        self.all_seed_hotsegment = np.empty(len(self.hotsegments), dtype=object)
        
        delete_indexes = []

        for route_index, seed_hotsegment in enumerate(self.hotsegments):
            
            self.all_seed_hotsegment[route_index] = seed_hotsegment

            if self.debug:
                print("-- Seed: {}, Created routes: {}".format(seed_hotsegment,
                                                               len(self.patrol_routes_list)))
            # add excluded seed hotsegments to check neighboring
            check_hotsegments = deepcopy(self.hotsegments)
            patrol_subgraph = self.build_route(seed_hotsegment, check_hotsegments, 
                                               min_length, max_length)

            if patrol_subgraph is not False:
                self.all_routes_subgraph[route_index] = patrol_subgraph
                num_events = patrol_subgraph.size(weight='score')
                self.all_routes_score[route_index] = num_events
            else:
                delete_indexes.append(route_index)
        
        self._delete_route_candidates(delete_indexes)
        
        # choose the best routes
        while len(self.patrol_routes_list) < num_routes:
            
            # Take the route with high score
            patrol_route_index = np.argmax(self.all_routes_score)
            high_patrol_subgraph = self.all_routes_subgraph[patrol_route_index] 
            seed_hotsegment = self.all_seed_hotsegment[patrol_route_index]
            
            self._delete_route_candidates(patrol_route_index)

            self.patrol_routes_list.append(high_patrol_subgraph)
            self._exclude_hotsegments(high_patrol_subgraph)
            self.graph.remove_edges_from(high_patrol_subgraph.edges)
            
            delete_counter = 0
            # verify if other hotsegment was included in the best graph, and delete it
            for route_index, check_route in enumerate(self.all_routes_subgraph):
                seed_hotsegment = self.all_seed_hotsegment[route_index - delete_counter]
                if (seed_hotsegment in high_patrol_subgraph.edges()) or (seed_hotsegment[::-1] in high_patrol_subgraph.edges()):
                    self._delete_route_candidates(route_index - delete_counter)
                    delete_counter += 1
                    continue

                intersect_edges = [edge for edge in high_patrol_subgraph.edges() 
                                   if edge in check_route.edges() or 
                                      edge[::-1] in check_route.edges()]
                if intersect_edges:
                    # add excluded seed hotsegments to check neighboring
                    check_hotsegments = deepcopy(self.hotsegments)
                    patrol_subgraph = self.build_route(seed_hotsegment, check_hotsegments, 
                                                       min_length, max_length)

                    if patrol_subgraph is not False:
                        self.all_routes_subgraph[route_index - delete_counter] = patrol_subgraph
                        num_events = patrol_subgraph.size(weight='score')
                        self.all_routes_score[route_index - delete_counter] = num_events
                    else:
                        self._delete_route_candidates(route_index - delete_counter)
                        delete_counter += 1
        
        if direct_path:
            self._add_direct_path()

        return self
    
    def _add_direct_path(self):
        for index, patrol_subgraph in enumerate(self.patrol_routes_list):

            check_hotsegments = deepcopy(self.hotsegments)
            new_patrol = self.add_direct_path(self.graph, patrol_subgraph, 
                                              check_hotsegments, self.max_length)
            if new_patrol is not False:
                self.patrol_routes_list[index] = new_patrol
                self._exclude_hotsegments(new_patrol)
        
    def build_route(self, seed_hotsegment, hotsegments, min_length, max_length):
        
        if seed_hotsegment in hotsegments:
            hotsegments.remove(seed_hotsegment)
        
        root_subgraph = create_root_subgraph(self.graph, seed_hotsegment, max_length)
        root_subgraph = nx.Graph(root_subgraph)
        
        if self.debug:
            print("Root subgraph")
            self.plot_patrol(root_subgraph)

        # create the initial patrol subgraph and add the seed
        patrol_subgraph = nx.Graph()
        add_edge_subgraph(root_subgraph, patrol_subgraph, seed_hotsegment)
        patrol_route_length = root_subgraph.edges[seed_hotsegment][self.weight]
        # remove the seed from the patrol
        root_subgraph.remove_edge(*seed_hotsegment)

        neighb_hotsegments = [edge for edge in root_subgraph.edges() 
                              if (edge in hotsegments) or (edge[::-1] in hotsegments)]
        
        # save all hotsegments from root subgraph 
        root_hotsegments = [seed_hotsegment] + neighb_hotsegments

        while patrol_route_length < max_length:

            distance_available = max_length - patrol_route_length

            selected_path = min_hotsegment_cycle_path(root_subgraph, patrol_subgraph, root_hotsegments, 
                                                      neighb_hotsegments, distance_available, debug=self.debug)
            if not bool(selected_path):
                break

            add_path = selected_path['add_path']
            add_path_edges = [(u,v) for u, v in zip(add_path, add_path[1:])]
            # exclude hotsegments that are part of the route
            neighb_hotsegments = [(u,v) for u, v in neighb_hotsegments 
                                  if ((u,v) not in add_path_edges) and
                                     ((v,u) not in add_path_edges)]   

            # remove edges from root subgraph
            root_subgraph.remove_edges_from(patrol_subgraph.edges)

            patrol_route_length = selected_path['patrol_route_length']
            patrol_subgraph = selected_path['patrol_subgraph']
            
            if self.debug:
                print("Round trip length", round_trip_size(patrol_subgraph))
                #print("Percentage of cycle", percentage_route_cycle(patrol_subgraph))
                self.plot_patrol(patrol_subgraph)
                
        patrol_route_length = round_trip_size(patrol_subgraph)

        if (patrol_route_length >= min_length) and (patrol_route_length <= max_length):
            return patrol_subgraph
        else:
            if self.debug:
                print("-X- Patrol Route not added")
            return False
        
    def add_direct_path(self, graph, patrol_subgraph, hotsegments, max_length):

        patrol_route_length = round_trip_size(patrol_subgraph)
        modified_route = False
        
        while patrol_route_length < max_length:

            distance_available = max_length - patrol_route_length

            selected_path = min_hotsegment_path(graph, patrol_subgraph, hotsegments, 
                                                distance_available, debug=self.debug)
            if not bool(selected_path):
                break
            else:
                modified_route = True

            add_path = selected_path['add_path']
            add_path_edges = [(u,v) for u, v in zip(add_path, add_path[1:])]
            # exclude hotsegments that are part of the route
            hotsegments = [(u,v) for u, v in hotsegments 
                           if ((u,v) not in add_path_edges) and
                              ((v,u) not in add_path_edges)]  

            # remove edges from root subgraph
            graph.remove_edges_from(patrol_subgraph.edges)

            patrol_route_length = selected_path['patrol_route_length']
            patrol_subgraph = selected_path['patrol_subgraph']
            
            if self.debug:
                print("Round trip length", round_trip_size(patrol_subgraph))
                #print("Percentage of cycle", percentage_route_cycle(patrol_subgraph))
                self.plot_patrol(patrol_subgraph)
        
        if modified_route:
            return patrol_subgraph
        else:
            return False
        
    def plot_patrol(self, patrol_subgraph):
        
        plot_graph = nx.MultiDiGraph(patrol_subgraph)
        plot_graph.add_nodes_from((n, self.graph.nodes[n]) for n in plot_graph.nodes)
        plot_graph.graph = self.graph.graph
        ox.plot_graph(plot_graph)
        plt.show()
