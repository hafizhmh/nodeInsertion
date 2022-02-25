from math import sqrt
from platform import node
from turtle import pos
import numpy
import xmltodict
import json
import pandas
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from functools import cache, lru_cache

x_coord = "@xCoord"
y_coord = "@yCoord"

def n2p_to_json(filename):
    """
    Convert n2p file (xml) to json data

    Also creating json dump file
    """
    with open(filename) as f:
        n2p_dict = xmltodict.parse(f.read())
    f.close()
    json_data = json.dumps(n2p_dict)
    with open(f"{filename}.json", "w") as f:
            f.write(json_data)
    f.close()
    return n2p_dict

def node_extractor(n2p_dict):
    """
    Extract node information from n2p json data

    Return a list of dicts, each dict has these keys:
    @id, @xCoord, @yCoord, @name
    """
    try:
        nodes = n2p_dict["network"]["node"]
    except KeyError as e:
        try:
            nodes = n2p_dict["network"]["physicalTopology"]["node"]
        except Exception as e:
            print(e)
    return nodes


def calc_centroid(nodes_list):
    """
    Return tuple (x, y) as average coordinate from the list of nodes
    """
    x_sum = 0
    y_sum = 0
    for node in nodes_list:
        y_sum += float(node[x_coord])
        x_sum += float(node[y_coord])
    return (x_sum/len(nodes_list),y_sum/len(nodes_list))


def distance_sorter(nodes, center=None, sort=None, reverse=None):
    """
    Sort nodes according to its distance to the center.

    Return list of string of node names.

    Do no sorting when sort == False.
    When sort == True, return sorted list of nodes
    from 'furthest' to the 'nearest' nodes to center,
    vice versa.
    """
    if sort == None:
        sort = True
    if reverse == None:
        reverse = True
    if sort:
        node_distance = []
        for node in nodes:
            print(node) if debug else None
            print(node["@id"]) if debug else None
            x = float(node[x_coord])
            y = float(node[y_coord])
            dis = sqrt((x-center[0])**2 +(y-center[1])**2)
            node_distance.append((node["@id"], dis))
        node_distance.sort(key=lambda x:x[1],reverse = reverse)
        sorted_node =  [node for (node,distance) in node_distance]
    else:
        sorted_node =  [node["@id"] for node in nodes]
    return sorted_node

def gen_cost_matrix(nodes):
    # prepare for the costs matrix with empty array.
    costs = numpy.empty([len(nodes), len(nodes)])
    i = 0
    for node1 in nodes:
        j = 0
        for node2 in nodes:
            x1 = float(node1[x_coord])
            y1 = float(node1[y_coord])
            x2 = float(node2[x_coord])
            y2 = float(node2[y_coord])
            # cost = euclidian distance between two nodes
            dis = sqrt((x1-x2)**2 +(y1-y2)**2)
            costs[i][j] = dis
            j+=1
        i+=1
    return costs

# def node_insertion(size,nodes,df):
def node_insertion(size,center,sort=None,reverse=None):
    # sort node according to the distance from center
    nodes_sliced = nodes[:size]
    sorted_node = distance_sorter(nodes=nodes_sliced, center=center, sort=sort, reverse=reverse)
    new_node = sorted_node[-1] # last item
    if size==1:
        return 0, [new_node]
        # stop function
    else:
        # recursive call, should be in cache
        last_cost, current_string = node_insertion(size-1,center)
        print("-----------")
        print("size", size)
        # generating possible rings
        try:
            del min_cost
        except Exception:
            pass
        for i in range (1,size):
            possible_ring = current_string[:]
            print("pre", possible_ring)
            possible_ring.insert(i,str(new_node))
            print("inserted", possible_ring)
            j = (i-1) % len(possible_ring)
            k = (i+1) % len(possible_ring)
            new_node = int(possible_ring[i])
            pre_node = int(possible_ring[j])
            post_node = int(possible_ring[k])
            print("neigh", pre_node,new_node,post_node)
            cost = last_cost - df[pre_node][post_node] + df[new_node][pre_node] + df[new_node][post_node]
            print("last_cost", last_cost)
            print("cost", cost)
            input()
            try:
                if cost<min_cost:
                    min_cost = cost
                    min_str = possible_ring
                    print()
            except Exception:
                # set the minimum cost = current cost
                # if the min_cost variable doesn't exist
                # (for the first comparison, instead of
                # assigning min_cost = inf befor the first loop)
                min_cost = cost
                min_str = possible_ring
    # preparing for next iteration
    current_string = min_str
    # reporting
    print("return", size, min_cost, min_str)
    return min_cost, min_str

def main():
    # set debug = True to print a bunch of detailed information
    global debug
    debug = False
    # choose the file to be read here
    # filename = 'ATTWorldNet_N90_E274.n2p'
    filename = '20 node.n2p'

    # open the n2p file that contains geographical locations
    n2p_dict = n2p_to_json(filename=filename)

    # extract only the relevant info
    global nodes
    nodes = node_extractor(n2p_dict=n2p_dict)

    # calculate distance between every two nodes, so #nodes**2 calculation.
    costs_matrix = gen_cost_matrix(nodes=nodes)

    # print the costs matrix
    global df
    df = pandas.DataFrame(costs_matrix, columns=[x for x in range(0,len(nodes))])
    df_rounded = df.round(decimals=3)
    print("Costs matrix (Euclidian distance):\n", df_rounded,"\n")
    print("--------------------------")
    input("Press enter to start the algorithm:\n")

    result_lst = []
    node_list = distance_sorter(nodes, center=None, sort=False)
    # calculate center of the nodes
    center = calc_centroid(nodes_list=nodes)
    # for size in tqdm(range(1,len(nodes)+1)):
    # for size in tqdm(range(1,len(node_list)+1)):
    for size in range(1,len(node_list)+1):
        start_time = time.perf_counter()
        min_cost, min_str = node_insertion(size=size,center=center,sort=False,reverse=False)
        duration = time.perf_counter() - start_time
        result = {"size": size, "min_cost": min_cost, "min_str":min_str, "duration": duration}
        result_lst.append(result)

    for result in result_lst:
        print(result["size"], result["min_cost"], result["duration"])

if __name__ == "__main__":
    main()



# function (size, list nodes, cost matrix)
# return cost, ring