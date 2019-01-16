function E_V = Exp_Value(V,y_pos)
global Order N Y_num piy Mesh_B Mesh_Bp Mesh_D
V_new = NaN(N*N,Y_num);
for i=1:Y_num
fun = cheby_interp_2d(Order,V(:,i),Mesh_B(:,1),Mesh_D(:,1));
dp = zeros(N*N,1);
bp = reshape(Mesh_Bp,N*N,1);
V_new(:,i) = fun(bp,dp);
end
Trans = piy(y_pos,:);
E_V = V_new*Trans';
E_V = reshape(E_V,N,N);
end
