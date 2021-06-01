# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021

"""
This is the main neighbor point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)

# Acknowledgment -> https://github.com/atakanozyapici/ECE448/blob/master/mp1-code/search.py

import queue
import copy


def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """

    start = maze.start
    waypoints = maze.waypoints
    frontier = queue.Queue()

    path = []
    visited = {}
    prev = {}
    path_nodes = []

    waypoints = list(waypoints)
    frontier.put(start)
    visited[start] = start

    while frontier.empty() == False:
        current = frontier.get()

        for neighbor in maze.neighbors(current[0], current[1]):
            flag = False

            for w in waypoints:
                if neighbor == w:
                    flag = True
                    path.insert(0, waypoints[0])
                    path_nodes = current
                    break 
            

            if (neighbor not in visited):
                frontier.put(neighbor)
                visited[neighbor] = neighbor
                prev[neighbor] = current

            if flag:
                break

        if flag:
            break

    if flag:

        while path_nodes != start:      
            path.insert(0, path_nodes)
            path_nodes = prev[path_nodes]

        path.insert(0, start)
        return path 

    else:
        path = [start]
        return path

    

def astar_single(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """

    start = maze.start
    waypoints = maze.waypoints
    frontier = queue.Queue()
    path = []
    visited = {}
    prev = {}
    heuristic = 0

    waypoints = list(waypoints)
    frontier.put((0,start))
    visited[start] = (0,start)

    while frontier.empty() == False:
        current = frontier.get()

        for neighbor in maze.neighbors(current[1][0], current[1][1]):
            flag = False

            for w in waypoints:
                if neighbor == w:
                    flag = True
                    break
            

            if (neighbor not in visited):
                heuristic = abs(waypoints[0][0] - neighbor[0]) + abs(waypoints[0][1] - neighbor[1])
                frontier.put((heuristic, neighbor))
                visited[neighbor] = neighbor
                prev[neighbor] = current[1]

            if flag:
                path.append(waypoints[0])
                path_nodes = current[1]

                while path_nodes != start:
                    
                    path.insert(0, path_nodes)
                    path_nodes = prev[path_nodes]

                path.insert(0, start)
                return path


    return path

def astar_corner(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are four corner objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
        """

    start = maze.start
    waypoints = maze.waypoints
    frontier = queue.PriorityQueue()

    path = []
    visited = {}
    prev = {}
    heuristic = 0

    # State format -> (f(n), g(n), (maze cell, (waypoints left to visit)))
    frontier.put((0, 0, (start, waypoints)))
    visited[(start, waypoints)] = (0, 0, (start, waypoints))

    while frontier.empty() == False:
        current = frontier.get()
        flag = False
        waypoints_left = current[2][1]

        if len(waypoints_left) == 0:
            flag = True
            # path.append(current[2])
            path_states = current[2]
            break

        for neighbor in maze.neighbors(current[2][0][0], current[2][0][1]):
            cur_waypoints = copy.deepcopy(waypoints_left)

            if (neighbor in cur_waypoints):
                cur_waypoints = list(cur_waypoints)
                cur_waypoints.remove(neighbor)
                cur_waypoints = tuple(cur_waypoints)
            

            if ((neighbor, cur_waypoints) not in visited):
                heuristic = 0

                if maze.size.x > 10:
                    for i in cur_waypoints:
                        heuristic += abs(neighbor[0] - i[0]) + abs(neighbor[1] - i[1])

                f_estimate = heuristic + (current[1] + 1)

                frontier.put((f_estimate, current[1] + 1, (neighbor, cur_waypoints)))
                visited[(neighbor, cur_waypoints)] = neighbor
                prev[(neighbor, cur_waypoints)] = current

    if flag:

        while (path_states != (start, waypoints)):

            path.insert(0, path_states[0])
            path_states = prev[path_states][2]

        path.insert(0, start)
        return path

    else:
        path = [start]
        return path



def astar_multiple(maze):
    """
    Runs A star for part 4 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    start = maze.start
    waypoints = maze.waypoints
    frontier = queue.PriorityQueue()

    path = []
    visited = {}
    prev = {}
    heuristic = []

    edge_weights = {}
    MST_lengths = {}

    # State format -> (f(n), g(n), (maze cell, (waypoints left to visit)))
    frontier.put((0, 0, (start, waypoints)))
    visited[(start, waypoints)] = (0, 0, (start, waypoints))

    # Getting edge weights for the maze and waypoints
    temp_maze = copy.deepcopy(maze)

    for i in waypoints:

        for j in waypoints:

            temp_maze.start = i
            temp_maze.waypoints = [j]
            edge_weights[(i, j)] = len(astar_single(temp_maze))


    while frontier.empty() == False:
        current = frontier.get()
        flag = False
        waypoints_left = current[2][1]

        if len(waypoints_left) == 0:
            flag = True
            path_states = current[2]
            break


        MST_cost = 0

        if waypoints_left in MST_lengths:
            MST_cost = MST_lengths[waypoints_left]

        else:
            MST_cost = get_MST_length(copy.deepcopy(waypoints_left), maze, edge_weights)
            MST_lengths[waypoints_left] = MST_cost


        for neighbor in maze.neighbors(current[2][0][0], current[2][0][1]):
            cur_waypoints = copy.deepcopy(waypoints_left)

            if (neighbor in cur_waypoints):
                cur_waypoints = list(cur_waypoints)
                cur_waypoints.remove(neighbor)
                cur_waypoints = tuple(cur_waypoints)
            

            if ((neighbor, cur_waypoints) not in visited):

                heuristic = []

                # if maze.size.x > 10:
                #     for i in cur_waypoints:
                #         heuristic += abs(neighbor[0] - i[0]) + abs(neighbor[1] - i[1])

                for i in waypoints_left:
                    heuristic.append(abs(neighbor[0] - i[0]) + abs(neighbor[1] - i[1]) + MST_cost)


                f_estimate = min(heuristic) + (current[1] + 1)

                frontier.put((f_estimate, current[1] + 1, (neighbor, cur_waypoints)))
                visited[(neighbor, cur_waypoints)] = neighbor
                prev[(neighbor, cur_waypoints)] = current

    if flag:

        while (path_states != (start, waypoints)):

            path.insert(0, path_states[0])
            path_states = prev[path_states][2]

        path.insert(0, start)
        return path

    else:
        path = [start]
        return path


def get_MST_length(waypoints_left, maze, edge_weights):

    cost = 0
    waypoints_found = []

    current = waypoints_left[0]
    Max = maze.size.x + maze.size.y

    while len(waypoints_left) != 0:

        waypoints_left = list(waypoints_left)
        waypoints_left.remove(current)
        waypoints_left = tuple(waypoints_left)

        waypoints_found.append(current)
        path_length = Max

        for i in waypoints_found:

            for j in waypoints_left:

                if edge_weights[(i,j)] < path_length:
                    path_length = edge_weights[(i,j)]
                    current = j

        cost += path_length  
        
    return cost


def fast(maze):
    """
    Runs suboptimal search algorithm for part 5.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    return []
    
            
