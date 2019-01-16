function [Vns_new,pos] = update_Vns(V_old)
global N Y_num Mesh_B Mesh_Bp Mesh_Dp y_grid q betta
Vns_new = NaN(N,Y_num);
pos = NaN(N,Y_num);
for i = 1:Y_num
    Q = repmat(q(:,i)',N,1);
    E_V = Exp_Value(V_old,i);
    X = -inf*ones(N,N);
    Y = U(y_grid(i)+Q.*Mesh_Bp-Mesh_B) + betta*E_V;
    X(Mesh_Dp==0)=Y(Mesh_Dp==0);
    [Vi, pos_i] = max(X,[],2);
    pos(:,i) = pos_i;
    Vns_new(:,i) = Vi;
end
end