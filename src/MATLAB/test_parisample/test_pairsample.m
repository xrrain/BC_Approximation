clc
clear
[graphs_matrix, graphs_name] = read_graphs;
graph_num = length(graphs_name);
remained_graph = 1:graph_num;
graph_types = ["directed", "undirected"];
graph_type = 0;
sample_size = 1000;
frac = 0.2;
result_nodes = [];
tocs = zeros(1,graph_num);
tods = zeros(1,graph_num);
todds = zeros(1,graph_num);
lengths = zeros(1,graph_num);
intervals = zeros(1,graph_num);
for index = 1:graph_num
    % most pairs of the 9th graph are not connected, so jump it.
    % the 10 and 18 th graph contains a negative cycle of length -58.1395, so jump it.
    tic
    if index == 9 || index==10 || index==18
        continue
    end
    graph_matrix = cell2mat(graphs_matrix(index));
    graph_name = string(graphs_name(index));
    if isdiag(graph_matrix)
        G = graph(graph_matrix);
    else 
        G = digraph(graph_matrix);
    end
    % simplify the graph by deleting the multi edge
    if ismultigraph(G)
        G = simplify(G);
    end
    all_nodes = 1: height(G.Nodes);
    num_nodes = length(all_nodes);
    all_pairs = log(num_nodes)^2*sqrt(num_nodes);
    iter_number = ceil(all_pairs/sample_size) + 1;
    p_nodes = ones(1,num_nodes)./num_nodes;
    seed = rng;
    target_node = get_targetnodes(G,frac,sample_size,all_nodes, p_nodes);
    % change the possiblity accoring to the selected note
    for iter = 1:iter_number
        p_pre = zeros(num_nodes);
        for node = target_node
            dist = distances(G,node);
            p_pre = process_p_far_max(dist,p_pre);
        end
        p_pre = p_pre / sum(p_pre);
        target_node = get_targetnodes(G,frac,sample_size,all_nodes, p_pre);
    end
    toc;
    intervals(index) = toc;
    wbc = centrality(G,'betweenness','Cost',G.Edges.Weight);
    fprintf("finish compute betweenness for number %d graph: %s\n",index, graph_name);
    normalized_wbc = wbc./((num_nodes-2)*(num_nodes-1));
    target_length = length(target_node);
    lengths(index) = target_length;
    [B,I] = maxk(normalized_wbc, target_length);
    toc = length(intersect(I,target_node))/target_length;
    tocs(index) = toc;
    tod = sum(normalized_wbc(target_node))/sum(normalized_wbc);
    tods(index) = tod;
    todd = sum(normalized_wbc(target_node))/sum(B);
    todds(index) = todd;
end

figure
p = plot(1:graph_num, tocs, 1:graph_num, tods,1:graph_num, todds);
legend()
