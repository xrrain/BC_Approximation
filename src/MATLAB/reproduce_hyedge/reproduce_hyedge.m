clc
clear
[graphs_matrix, graphs_name] = read_graphs("./graph_folder_mat/");
graph_num = length(graphs_name);
remained_graph = 1:graph_num;
graph_types = ["directed", "undirected"];
graph_type = 0;
result_nodes = [];
times=zeros(graph_num*3,1);
ex_r = zeros(graph_num*3,1);
sample_size_hyedge = zeros(graph_num*3,1);
hedge_r = zeros(graph_num*3,1);
speed_up = zeros(graph_num*3,1);
sample_size_yalg = zeros(graph_num*3,1);
yalg_r = zeros(graph_num*3,1);
yalg_speedup = zeros(graph_num*3,1);
col_names = {'time', 'exhaust_result', 'sample_size1', 'hyedge_result1', 'speedup1', 'sample_size2', 'hyedge_result2', 'speedup2'};
results_matrix = zeros(graph_num*3, 8);
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
    target_k = [10, 50, 100];
    eps = 0.1;
    all_pairs_hyedges = target_k.*log(num_nodes)./eps^2;
    all_pairs_yalg = 2*log(2*(num_nodes^3))./eps^2;
    p_nodes = ones(1,num_nodes)./num_nodes;
    seed = rng;
    for i = 1:length(target_k)
        tic;
        wbc = centrality(G,'betweenness');
        k = target_k(i);
        [B,~] = maxk(wbc, k);
        exhaust = sum(B)./(num_nodes*(num_nodes - 1));
        toc1 = toc;
        fprintf('the %d graph computing %d nodes time is: %f\n', index, k, toc1);
        results_matrix((index-1)*3 + i, 1) = toc1;
        times((index-1)*3 + i) = toc1;
        fprintf('the exhaust result is %f\n', exhaust);
        results_matrix((index-1)*3 + i, 2) = exhaust;
        ex_r((index-1)*3 + i)  = exhaust;
        tic;
        sample_size = all_pairs_hyedges(i);
        fprintf('hyedge sample size is %d\n' ,ceil(sample_size));
        results_matrix((index-1)*3 + i, 3) = ceil(sample_size);
        sample_size_hyedge((index-1)*3 + i)  = ceil(sample_size);
        target_nodes = get_targetnodes(G,k, ceil(sample_size),all_nodes, p_nodes);
        hedge = sum(wbc(target_nodes))/num_nodes/(num_nodes - 1);
        toc2 = toc;
        fprintf('the hyedge result is %f\n', hedge);
        results_matrix((index-1)*3 + i, 4) = hedge;
        hedge_r((index-1)*3 + i)  = hedge;
        fprintf('the speadup is %f\n', toc1/toc2);
        speed_up((index-1)*3 + i)  = toc1/toc2;
        results_matrix((index-1)*3 + i, 5) = toc1/toc2;
        tic;
        sample_size = all_pairs_yalg;
        fprintf('yalg sample size is %d\n' ,ceil(sample_size));
        target_nodes = get_targetnodes(G,k, ceil(sample_size),all_nodes, p_nodes);
        hedge = sum(wbc(target_nodes))./(num_nodes*(num_nodes - 1));
        toc3 = toc;
        fprintf('the hyedge result is %f\n', hedge);
        fprintf('the speadup is %f\n', toc1/toc3);
        sample_size_yalg((index-1)*3 + i)  = ceil(sample_size);
        yalg_r((index-1)*3 + i) = hedge;
        yalg_speedup((index-1)*3 + i)  = toc1/toc3;
        results_matrix((index-1)*3 + i, 6) = ceil(sample_size);
        results_matrix((index-1)*3 + i, 7) = hedge;
        results_matrix((index-1)*3 + i, 8) = toc1/toc3;
    end
end
save('results_matrix.mat', 'results_matrix');
p_table=table(times,ex_r,sample_size_hyedge,hedge_r,speed_up,sample_size_yalg,yalg_r,yalg_speedup,'VariableNames',col_names);
writetable(p_table,'result.xlsx');
