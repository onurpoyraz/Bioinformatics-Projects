import numpy as np

def de_bruijn(input_series):
    k = len(input_series[0])
    edges = []
    nodes = set()
    for item in input_series:
        edges.append((item, item[0 : k - 1], item[1 : k]))
        nodes.add(item[0 : k - 1])
        nodes.add(item[1 : k])
    return nodes, edges

def edge_finder(nodes, edges):
    founds = []
    io_counter = np.zeros((len(nodes), 2));
    for i, item in enumerate(nodes):
        for element in edges:
            if(element[1] == item):
                io_counter[i][0] = io_counter[i][0] +1
            if(element[2] == item):
                io_counter[i][1] = io_counter[i][1] +1
        if(io_counter[i][0] == 1 and io_counter[i][1] == 1):
            founds.append(item)
    return founds
 
def contig_generator(edges, founds):
    output_series = ''
    printed_series = ''
    edges_remaining = []
    for item in edges:
        if not [element for element in founds if element in item[0]]:
            output_series += '%s\n' % (item[0])
        else:
            edges_remaining.append(item)

    current_element = ''
    while (len(edges_remaining)>0):
        for item in edges_remaining:
            if (current_element == ''):
                if not [element for element in founds if element ==item[1]]:
                    output_series += '%s' % (item[0])
                    printed_series += '%s' % (item[0])
                    current_element = item[2]
                    edges_remaining.remove(item)
                    break;
            else:
                if (item[1] == current_element):
                    output_series += '%s' % (item[0][-1])
                    printed_series += '%s' % (item[0][-1])
                    current_element = item[2]
                    if not [element for element in founds if element == current_element]:
                        current_element = ''
                        output_series += '\n'
                        printed_series += '\n'
                    edges_remaining.remove(item)
                    break;
    return output_series, printed_series

input_series = []
with open('./test-input.txt') as f:
    input_series = f.read().splitlines() 

nodes, edges = de_bruijn(input_series)
founds = edge_finder(nodes, edges)
output_series, printed_series = contig_generator(edges, founds)
                
with open("test-output.txt", "w") as text_file:
    text_file.write(output_series)
with open("test-output-contigs-only.txt", "w") as text_file:
    text_file.write(printed_series)
    
print(printed_series)