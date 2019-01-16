function E_V = Exp_Value_S(V,y_pos)
global Order n N Y_num piy Mesh_B Mesh_D
V_new = NaN(n*N,Y_num);
for i=1:Y_num
%find bp in 2 steps 
bp = Mesh_B.*(1-Mesh_D); %I need to selct the diagonal of this matrix %current output not future --> c.f pdf
fun = cheby_interp(Order,V(:,i),Mesh_B(:,1));
bp = reshape(bp,N*n,1);
V_new(:,i) = fun(bp);
end
Trans = piy(y_pos,:);
E_V = V_new*Trans';
E_V = reshape(E_V,n,N);
end