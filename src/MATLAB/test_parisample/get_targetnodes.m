function target_node = get_targetnodes(G,frac, sample_size,all_nodes, p_nodes)
%GET_TARGETNODES Summary of this function goes here
%   Detailed explanation goes here
    sample_pairs = zeros(sample_size,2);
    for i = 1:sample_size
        src = sample_one(all_nodes, p_nodes);
        des = sample_one(all_nodes, p_nodes);
        src_pos = find(sample_pairs(:,1) == src);
        des_pos = find(sample_pairs(:,2) == des);
        while src == des || ~isempty(intersect(src_pos, des_pos))
            src = sample_one(all_nodes, p_nodes);
            des = sample_one(all_nodes, p_nodes);
            src_pos = find(sample_pairs(:,1) == src);
            des_pos = find(sample_pairs(:,2) == des);
        end
        sample_pairs(i,1) = src;
        sample_pairs(i,2) = des;
    end
    size_sample = size(sample_pairs);
    real_sample_size = size_sample(1);
    hyedges= [];
    distances= [];
    for i = 1:real_sample_size
        src = sample_pairs(i,1);
        des = sample_pairs(i,2);
        [P,d] = shortestpath(G,src,des); %it only return the first shortest path
        % but i cannot fina a suitable function to get all the shortest
        % path
        % if there are more than one paths, random select one
        if ~isempty(P)
            distances = [distances,d];
            P(1) = [];
            P(end) = [];
            hyedges = [hyedges,P];
        end
    end
    node_unique = unique(hyedges);
    if isempty(node_unique)
        return
    end
    [count,records] = histcounts(hyedges(:),[node_unique,inf]);
    freq = [node_unique',count'];
    % find the first number node account for xx% in all freq 
    [~,I] = sort(freq(:,2), 'descend');
    sum_freq = sum(freq(:,2));
    sum_selected = 0;
    target_node = [];
    for pos = 1:length(I)
        index = I(pos);
        target_node = [target_node, freq(index,1)];
        sum_selected = sum_selected + freq(index,2);
        if sum_selected / sum_freq >= frac
            break
        end
    end
end

