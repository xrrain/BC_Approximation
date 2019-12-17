function res = sample_one(data,p)
%sample_one Summary of this function goes here
%   Detailed explanation goes here
%   this function sample from data according to p,which
%   represent each item's posibility
p_sum = zeros(length(p));
p_sum(1) = p(1);
for i =2 : length(p_sum)
    p_sum(i) = p(i) + p_sum(i-1);
end
p_sample = rand(1);
all = find(p_sum >= p_sample);
pos = all(1);
res = data(pos);
end

