function [starts,stops] = convert_segments(state_vector,state)
    state_vector = state_vector';
    vec = [0,state_vector == state];
    starts = find(diff(sign(vec))>0);
    stops = find(diff(sign(vec))<0);
end

