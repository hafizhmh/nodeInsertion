from math import sqrt
from turtle import color
from flask import Blueprint
import numpy
import xmltodict
import json
import pandas
from itertools import permutations
import time
import matplotlib.pyplot as plt

def circular_perms(my_list):
    """
    function to generate cyclic permutation from a list

    Accept a list as argument, generate a list of list that describes 
    [0,1,2,3]" => [[0, 1, 2, 3], [0, 1, 3, 2], [0, 2, 1, 3], [0, 2, 3, 1], [0, 3, 1, 2], [0, 3, 2, 1]]
    """
    return [my_list[:1]+list(perm) for perm in permutations(my_list[1:])]

def main():
    # set debug = True to print a bunch of detailed information
    debug = False
    # choose the file to be read here
    # filename = 'ATTWorldNet_N90_E274.n2p'
    filename = '200 node.n2p'

    # open the n2p file that contains geographical locations
    with open(filename) as xml_file:
        data_dict = xmltodict.parse(xml_file.read())
    xml_file.close()
    json_data = json.dumps(data_dict)
    with open("json_dump.json", "w") as json_file:
            json_file.write(json_data)
    json_file.close()

    try:
        nodes = data_dict["network"]["node"]
    except KeyError as e:
        try:
            nodes = data_dict["network"]["physicalTopology"]["node"]
        except Exception as e:
            print(e)

    costs = numpy.empty([len(nodes), len(nodes)]) # prepare for the costs matrix with empty array

    #################### exp
    x_sum = 0
    y_sum = 0
    for node in nodes:
        y_sum += float(node["@xCoord"])
        x_sum += float(node["@yCoord"])
    center = (x_sum/len(nodes),y_sum/len(nodes))
    print(center)

    node_distance = []
    for node in nodes:
        print(node) if debug else None
        print(node["@id"]) if debug else None
        x = float(node["@xCoord"])
        y = float(node["@yCoord"])
        dis = sqrt((x-center[0])**2 +(y-center[1])**2)
        node_distance.append((node["@id"], dis))
    print(node_distance)
    node_distance.sort(key=lambda x:x[1],reverse = True)
    print(node_distance)

    # calculate distance between every two nodes, so #nodes**2 calculation
    i = 0
    for node1 in nodes:
        j = 0
        for node2 in nodes:
            x1 = float(node1["@xCoord"])
            y1 = float(node1["@yCoord"])
            x2 = float(node2["@xCoord"])
            y2 = float(node2["@yCoord"])
            # cost = euclidian distance between two nodes
            dis = sqrt((x1-x2)**2 +(y1-y2)**2)
            costs[i][j] = dis
            j+=1
        i+=1
    # print the costs matrix
    df = pandas.DataFrame(costs, columns=[x for x in range(0,len(nodes))])
    df_rounded = df.round(decimals=3)
    print("Costs matrix (Euclidian distance):\n", df_rounded,"\n")
    print("--------------------------")
    input("Press enter to start the algorithm:\n")

    x = []
    y = []
    z = []
    max_size = len(df)
    start_time = time.time()
    for i, (new_node,_) in enumerate(node_distance):
        size = i+1
        x.append(size)
        if size==1:
            possible_rings = [[str(new_node)]]
        else:
            iter_time = time.time()
            # generating possible rings
            possible_rings = []
            for i in range (1,size):
                possible_ring = current_string[:]
                possible_ring.insert(i,str(new_node))
                possible_rings.append(possible_ring)
        print("Possible rings:\n", possible_rings) if debug else None

        # try each of the possible ring
        try:
            del min_cost
        except Exception:
            pass
        # min_cost
        for string in possible_rings:
            print(string, end=" | ") if debug else None

            # cost calculation
            cost = 0
            for i in range(len(string)):
                j = i+1
                if j >= len(string):
                    j=0

                src = int(string[i])
                dst = int(string[j])
                cost += df[src][dst]
            print("cost: {:.4f}".format(cost), end=" | ") if debug else None
            # Update the minimum cost
            try:
                if cost<min_cost:
                    min_cost = cost
                    min_str = string
            except Exception as e:
                # set the minimum cost = current cost
                # if the min_cost variable doesn't exist
                # (for the first comparison, instead of
                # assigning min_cost = inf befor the first loop)
                min_cost = cost
                min_str = string
            print("current min cost: {}".format(min_cost)) if debug else None

        # preparing for next iteration
        current_string = min_str
        # reporting
        duration = time.time() - start_time
        y.append(duration)
        z.append(min_cost)
        # print("size: {:2}".format(size), end= " | ")
        # print("{}".format("["+",".join(min_str)+"]"))

        # print("size: {:2}".format(size), end= " | ")
        # print("min cost: {:>13.8f}".format(min_cost), end= " | ")
        # print(min_cost)
        # print(duration)
        # print("time from the start: {:>13.10f} seconds".format(duration))
        # print("best ring: {}".format(",".join(min_str)+"-"+min_str[0]))
        # print("--------------------------") if debug else None
        # input("Press enter\n") if debug else None

    (koef2, koef1, koef0) = numpy.polyfit(x, y,2)
    # print(koef2, koef1, koef0)
    y1 = [koef2 * x**2 + koef1 * x + koef0 for x in x]

    # plotting
    plt.figure(0)
    plt.title("Time")
    plt.xlabel("Ring size")
    plt.ylabel("Time (seconds)")
    plt.plot(x, y,color ="red",label="Durasi Aktual")
    plt.plot(x,y1,color="blue",label=fr'y=${koef2:.6f} x^2 + ${koef1:.6f} x + ${koef0:.6f}$')
    plt.legend()

    plt.figure(1)
    plt.title("Score")
    plt.xlabel("Ring size")
    plt.ylabel("Score")
    plt.plot(x, z, color ="red")
    plt.show()

if __name__ == "__main__":
    main()