function E_V = Exp_Value_S(V,y_pos)
global Order N Y_num piy Mesh_B Mesh_Dp Mesh_D 
V_new = NaN(N*N,Y_num);
for i=1:Y_num
bp = Mesh_B.*(1-Mesh_Dp);
fun = cheby_interp_2d(Order,V(:,i),Mesh_B(:,1),Mesh_D(:,1));
bp = reshape(bp,N*N,1);
dp = reshape(Mesh_Dp,N*N,1);
V_new(:,i) = fun(bp,dp);
end
Trans = piy(y_pos,:);
E_V = V_new*Trans';
E_V = reshape(E_V,N,N);
end