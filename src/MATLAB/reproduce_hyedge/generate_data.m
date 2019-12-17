clc
clear
[graphs_matrix, graphs_name] = read_graphs("./graph_folder_mat/");
graph_num = length(graphs_name);
remained_graph = 1:graph_num;
graph_types = ["directed", "undirected"];
graph_type = 0;
result_nodes = [];
max_errors = zeros(1, graph_num);
mean_errors = zeros(1, graph_num);
max_relative_errors = zeros(1, graph_num);
mean_relative_errors = zeros(1, graph_num);
max_distance = 1e4;
for index = 1:graph_num
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
    % sample beacons
    beacons_num = 100;
    eps = 0.1;
    % with beacons random chosen, nodes with high bc
    sample_size = 2*log(2*(num_nodes^3))./eps^2;
    p_nodes = ones(1,num_nodes)./num_nodes;
    beacons = get_targetnodes(G, beacons_num, ceil(sample_size), all_nodes, p_nodes);
    dist_info = distances(G,beacons);
    pairs_num = 1e5;
    pairs_step_num = 2000;
    all_src = zeros(1, pairs_num);
    all_des = zeros(1, pairs_num);
    step_num = ceil(pairs_num / pairs_step_num);
    remain_num = rem(pairs_num, pairs_step_num);
    for step_id = 1 : step_num
        srcs = randperm(num_nodes, pairs_step_num);
        all_src(1, (step_id-1)* pairs_step_num + 1: step_id*pairs_step_num) = srcs;
        dess = randperm(num_nodes, pairs_step_num);
        all_des(1, (step_id-1)* pairs_step_num + 1: step_id*pairs_step_num) = dess;
    end
    if remain_num ~= 0
        srcs = randperm(num_nodes, remain_num);
        all_src(1, pairs_num-remain_num+1: pairs_num) = srcs;
        dess = randperm(num_nodes, remain_num);
        all_des(1, pairs_num-remain_num+1: pairs_num) = dess;
    end
    dataset = zeros(pairs_num, 2*beacons_num + 1);
    dist_src = dist_info(:,all_src)';
    dist_des = dist_info(:,all_des)';
    dist_all_s = distances(G);
    dist_real = zeros(1, pairs_num);
    for pair_id = 1: pairs_num
        src = all_src(1, pair_id);
        des = all_des(1, pair_id);
        dist_real(1, pair_id) = dist_all_s(src, des);
    end
    dataset(:,1: beacons_num) = dist_src;
    dataset(:,beacons_num+1: 2*beacons_num) = dist_des;
    dataset(:, end) = dist_real';
    pairs = [all_src', all_des'];
    splited_names = strsplit(graph_name, '/');
    writematrix(dataset, splited_names(2)+ '_dataset.csv');
    writematrix(pairs, splited_names(2)+ '_pairinfo.csv');
    writematrix(beacons', splited_names(2)+ '_beaconsinfo.csv');
    fprintf('finish graph: ' + graph_name + '\n');
end
