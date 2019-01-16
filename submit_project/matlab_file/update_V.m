function [S,V_new,pos] = update_V(Vs,Vns,pos_s,pos_ns)
global n Y_num 
S = zeros(n,Y_num);
O = ones(n,Y_num);
S(Vs>Vns) = O(Vs>Vns);
V_new = Vns;
V_new(Vs>Vns) = Vs(Vs>Vns);
pos = pos_ns;
pos(Vs>Vns) = pos_s(Vs>Vns);
end