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
    eps = 0.1;
    beacons_num = 100;
    all_pairs_hyedges = beacons_num.*log(num_nodes)./eps^2;
    all_pairs_yalg = 2*log(2*(num_nodes^3))./eps^2;
    p_nodes = ones(1,num_nodes)./num_nodes;
    % with nodes random chosen, becons with high bc
    % sample beacons
    sample_size = all_pairs_yalg;
    beacons = get_targetnodes(G, beacons_num, ceil(sample_size), all_nodes, p_nodes);
    
    dist_info = distances(G,beacons);
    seed = rng;
    tic;
    wbc = centrality(G,'betweenness');
    toc1 = toc;
    fprintf('the %d graph computing time is: %f\n', index, toc1);
    tic;
    normalized_wbc = wbc./((num_nodes-2)*(num_nodes-1));
    fprintf('hyedge sample size is %d\n' ,ceil(sample_size));
    
    % get the target nodes to estimate BC
    % sample nodes
    node_num = 100;
    target_nodes = [];
    while length(target_nodes) < node_num
        node = floor(rand()*num_nodes) + 1;
        target_nodes = [target_nodes, node];
        target_nodes = unique(target_nodes);
    end
    real_bc = normalized_wbc(target_nodes)';
    num_pairs = ceil(all_pairs_hyedges);
    sample_pairs = zeros(num_pairs,2);
    for i = 1:num_pairs
        src = floor(rand()* num_nodes) + 1;
        des = floor(rand()* num_nodes) + 1;
        src_pos = find(sample_pairs(:,1) == src);
        des_pos = find(sample_pairs(:,2) == des);
        while src == des || ~isempty(intersect(src_pos, des_pos))
            src = floor(rand()*num_nodes) + 1;
            des = floor(rand()*num_nodes) + 1;
            src_pos = find(sample_pairs(:,1) == src);
            des_pos = find(sample_pairs(:,2) == des);
        end
        sample_pairs(i,1) = src;
        sample_pairs(i,2) = des;
    end
    node_pair_info = zeros(num_pairs, 1);
    node_pair_info_s = zeros(num_pairs, 1);
    for pair_id = 1:num_pairs
        src = sample_pairs(pair_id,1);
        des = sample_pairs(pair_id,2);
        becons_info = dist_info(:,src) + dist_info(:, des);
        becons_info_s = abs(dist_info(:, src) - dist_info(:, des));
        min_dist = min(becons_info);
        max_dist = max(becons_info_s);
        node_pair_info(pair_id, 1) = min_dist; % the dist according to the becons
        node_pair_info_s(pair_id, 1) = max_dist;
    end
%     node_estimation_info = zeros(length(target_nodes), num_pairs);
%     for node = target_nodes
%         node_info = distances(G, node);
%         srcs_dist = node_info(1, sample_pairs(:,1)');
%         dess_dist = node_info(1, sample_pairs(:,2)');
%         pairs_dist = srcs_dist + dess_dist;
%         node_estimation_info(node, :) = pairs_dist < node_pair_info';
%     end
    % another way to implement the above vectorized
    nodes_pair_info = repmat(node_pair_info',length(target_nodes),1);
    nodes_info = distances(G, target_nodes);
    srcs_dists = nodes_info(:, sample_pairs(:,1)');
    dess_dists = nodes_info(:, sample_pairs(:,2)');
    pairs_dists = srcs_dists + dess_dists;
    nodes_pair_info_s = repmat(node_pair_info_s',length(target_nodes),1);
    node_estimation_info = (pairs_dists <= nodes_pair_info & pairs_dists >= nodes_pair_info_s);
    
    node_estimation = sum(node_estimation_info, 2)'./num_pairs;
    errors = abs(real_bc - node_estimation);
    relative_errors = abs(real_bc - node_estimation)./real_bc;
    relative_errors(relative_errors == Inf) = [];
    max_errors(index) = max(errors);
    mean_errors(index) = mean(errors);
    max_relative_errors(index) = max(relative_errors);
    mean_relative_errors(index) = mean(relative_errors);
end
figure;
graph_list = 1: graph_num;
plot(graph_list, log(max_errors), graph_list, log(mean_errors))
legend('max error', 'mean error');
figure;
plot(graph_list, max_relative_errors, graph_list, mean_relative_errors)
legend('max relative error', 'mean relative error')
