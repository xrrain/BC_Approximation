function [graphs_matrix,graphs_name] = read_graphs(graphs_folder)
%READ_GRAPHS Summary of this function goes here
%   Detailed explanation goes here
if nargin == 0
    graphs_folder = "./graph_folder/";
end
graphs_list = dir(graphs_folder);
n = length(graphs_list);
graphs = [];
for i=1:n
    if strcmp(graphs_list(i).name,'.')==1||strcmp(graphs_list(i).name,'..')==1  
        continue;  
    else
        graphs = [graphs; graphs_list(i)];
    end
end
num_graph = length(graphs);
graphs_matrix = cell(num_graph,1);
graphs_name = cell(num_graph,1);
[~,index]=sort([graphs.bytes]);
sorted_graph = graphs(index); %% from the first small start
for i = 1:num_graph
    graph_struct = load(graphs_folder + sorted_graph(i).name);
    graph_name = graph_struct.Problem.name;
    graph_matrix = graph_struct.Problem.A;
    graphs_name(i) = {graph_name};
    graphs_matrix(i) = {graph_matrix};
end
fprintf("finish read %d graphs\n", num_graph);
end

