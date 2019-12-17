function target_node = get_targetnodes(G,k, sample_size,all_nodes, p_nodes)
%GET_TARGETNODES Summary of this function goes here
%   Detailed explanation goes here
%     sample_pairs = zeros(sample_size,2);
%     for i = 1:sample_size
%         src = floor(rand()*length(all_nodes)) + 1;
%         des = floor(rand()*length(all_nodes)) + 1;
%         src_pos = find(sample_pairs(:,1) == src);
%         des_pos = find(sample_pairs(:,2) == des);
%         while src == des || ~isempty(intersect(src_pos, des_pos))
%             src = floor(rand()*length(all_nodes)) + 1;
%             des = floor(rand()*length(all_nodes)) + 1;
%             src_pos = find(sample_pairs(:,1) == src);
%             des_pos = find(sample_pairs(:,2) == des);
%         end
%         sample_pairs(i,1) = src;
%         sample_pairs(i,2) = des;
%     end
    num_nodes = length(all_nodes);
    sample_pairs = zeros(sample_size, 2);
    pairs_step_num = 1000;
    pairs_num = sample_size;
    step_num = ceil(pairs_num / pairs_step_num);
    remain_num = rem(pairs_num, pairs_step_num);
    for step_id = 1 : step_num
        srcs = randperm(num_nodes, pairs_step_num);
        sample_pairs((step_id-1)* pairs_step_num + 1: step_id*pairs_step_num, 1) = srcs;
        dess = randperm(num_nodes, pairs_step_num);
        sample_pairs((step_id-1)* pairs_step_num + 1: step_id*pairs_step_num, 2) = dess;
    end
    if remain_num ~= 0
        srcs = randperm(num_nodes, remain_num);
        sample_pairs(pairs_num-remain_num+1: pairs_num, 1) = srcs;
        dess = randperm(num_nodes, remain_num);
        sample_pairs(pairs_num-remain_num+1: pairs_num, 2) = dess;
    end
    real_sample_size = sample_size;
    hyedges= [];
    for i = 1:real_sample_size
        src = sample_pairs(i,1);
        des = sample_pairs(i,2);
        [P,~] = shortestpath(G,src,des); %it only return the first shortest path
        % but i cannot fina a suitable function to get all the shortest
        % path
        % if there are more than one paths, random select one
        n_p = length(P);
        if ~isempty(P) && n_p >2
            P(1) = [];
            P(n_p-1) = [];
            hyedges = [hyedges,P];
        end
    end
    node_unique = unique(hyedges);
    if isempty(node_unique)
        return
    end
    [count,~] = histcounts(hyedges(:),[node_unique,inf]);
    freq = [node_unique',count'];
    % find the first number node account for xx% in all freq 
    [~,I] = sort(freq(:,2), 'descend');
    target_node = [];
    for pos = 1:length(I)
        index = I(pos);
        target_node = [target_node, freq(index,1)];
        if length(target_node) >= k
            break
        end
    end
end

