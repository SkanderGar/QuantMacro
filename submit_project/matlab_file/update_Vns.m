function [Vns_new,pos] = update_Vns(V_old)
global n N Y_num Mesh_B Mesh_Bp Mesh_D y_grid q betta
Vns_new = NaN(n,Y_num);
pos = NaN(n,Y_num);
for i = 1:Y_num
    Q = repmat(q(:,i)',n,n);
    E_V = Exp_Value(V_old,i);
    X = -inf*ones(n,N);
    Y = U(y_grid(i)+Q.*Mesh_Bp-Mesh_B) + betta*E_V;
    X(Mesh_D==0)=Y(Mesh_D==0);
    [Vi, pos_i] = max(X,[],2);
    pos(:,i) = pos_i;
    Vns_new(:,i) = Vi;
end
end