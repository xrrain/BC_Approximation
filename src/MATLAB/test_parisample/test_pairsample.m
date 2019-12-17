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

tocs_c = zeros(1,graph_num);
tods_c = zeros(1,graph_num);
todds_c = zeros(1,graph_num);
lengths_c = zeros(1,graph_num);

toc1s = zeros(1,graph_num);
tod1s = zeros(1,graph_num);
todd1s = zeros(1,graph_num);


for index = 1:graph_num
    % most pairs of the 9th graph are not connected, so jump it.
    % the 10 and 18 th graph contains a negative cycle of length -58.1395, so jump it.
    tic
    if index == 9 || index==10 || index==18
        continue
    end
    graph_matrix = cell2mat(graphs_matrix(index));
    graph_name = string(graphs_name(index));
    if istril(graph_matrix) || istriu(graph_matrix) ||  issymmetric(graph_matrix)
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
    target_node1 = get_targetnodes(G,frac,sample_size,all_nodes, p_nodes);
    target_node = target_node1;
    compared_node = get_targetnodes(G, frac, ceil(all_pairs),all_nodes, p_nodes);
    % change the possiblity accoring to the selected note
    for iter = 1:iter_number
        p_pre = zeros(1,num_nodes);
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
    compared_length = length(compared_node);
    target1_length = length(target_node1);
    lengths(index) = target_length;
    [B,I] = maxk(normalized_wbc, target_length);
    [B_1,I_1] = maxk(normalized_wbc, target1_length);
    
    toc = length(intersect(I,target_node))/target_length;
    
    [B_C,I_C] = maxk(normalized_wbc, compared_length);
    toc_c = length(intersect(I_C,compared_node))/compared_length;
    
    toc1 = length(intersect(I_1,target_node1))/target1_length;
    
    tocs(index) = toc;
    tocs_c(index) = toc_c;
    toc1s(index) = toc1;
    
    tod = sum(normalized_wbc(target_node))/sum(normalized_wbc);
    tod_c = sum(normalized_wbc(compared_node))/sum(normalized_wbc);
    tod1 = sum(normalized_wbc(target_node1))/sum(normalized_wbc);
    tods(index) = tod;
    tods_c(index) = tod_c;
    tod1s(index) = tod1;
    
    todd = sum(normalized_wbc(target_node))/sum(B);
    todd_c = sum(normalized_wbc(compared_node))/sum(B_C);
    todd1 = sum(normalized_wbc(target_node1))/sum(B_1);
    
    todds(index) = todd;
    todds_c(index) = todd_c;
    todd1s(index) = todd1;
end

figure
p = plot(1:graph_num, tocs, 1:graph_num, toc1s,1:graph_num, tocs_c);
legend()

figure
p = plot(1:graph_num, tods, 1:graph_num, tod1s,1:graph_num, tods_c);
legend()

figure
p = plot(1:graph_num, todds, 1:graph_num, todd1s,1:graph_num, todds_c);
legend()
