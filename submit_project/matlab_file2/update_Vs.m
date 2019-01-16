function [Vs_new,pos] = update_Vs(Vs_old,Vns_old)
global N Y_num Mesh_D Mesh_Dp y_grid lamda tau betta omega1 omega2
Vs_new = NaN(N,Y_num);
pos = NaN(N,Y_num);
for i = 1:Y_num   
F = omega1*Mesh_Dp + omega2*Mesh_D;
E_V = Exp_Value_S(Vns_old,i);
E_V_S = Exp_Value_S(Vs_old,i);
X = U(y_grid(i)*(1-tau(i)*F));
X = X + betta*(lamda*F.*E_V_S+(1-lamda*F).*E_V);
[Vi, pos_i] = max(X,[],2);
Vs_new(:,i) = Vi;
pos(:,i) = pos_i;
end
end