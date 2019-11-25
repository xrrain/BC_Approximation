function p_post = process_p_far_max(dist, p_pre)
%process_p_far_max Summary of this function goes here
%   the function process the node with dist to change the p_pre
if length(dist)~=length(p_pre)
    p_post= p_pre;
    return
end
removed_inf = dist(~isinf(dist));
max_distance = max(removed_inf);
min_distance = min(removed_inf);

p = (dist - min_distance)/max_distance;
p(isinf(p))= 0;
p_post = p_pre + p;
end

