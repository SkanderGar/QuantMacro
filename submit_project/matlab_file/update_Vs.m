function [Vs_new,pos] = update_Vs(Vs_old,Vns_old)
global n Y_num Mesh_D y_grid lamda tau betta
Vs_new = NaN(n,Y_num);
pos = NaN(n,Y_num);
for i = 1:Y_num    
E_V = Exp_Value_S(Vns_old,i);
E_V_S = Exp_Value_S(Vs_old,i);
X = U(y_grid(i)*(1-tau(i)*Mesh_D));
X = X + betta*(lamda*Mesh_D.*E_V_S+(1-lamda*Mesh_D).*E_V);
[Vi, pos_i] = max(X,[],2);
Vs_new(:,i) = Vi;
pos(:,i) = pos_i;
end
end