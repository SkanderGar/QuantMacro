function [S,V_new,Vs_old,Vns_old,Pos,g_b,g_d] = VFI(Tol, Vs_start, Vns_start)
global N Y_num Mesh_Dp Mesh_Bp
err=1;
Vs_old = Vs_start;
Vns_old = Vns_start;
V_old = Vs_start;
j = 1;
while err>Tol
    [Vns_new,pos_ns] = update_Vns(Vns_old);
    [Vs_new,pos_s] = update_Vs(Vs_old,Vns_old);
    [S,V_new,Pos] = update_V(Vs_new,Vns_new,pos_s,pos_ns);
    err = max(max(abs(V_old-V_new)));
    if mod(j,4)==1
        fprintf('        Itereation: %d \n',j);
        fprintf('        Error: %d \n',err);
    end
    Vs_old = Vs_new;
    Vns_old = Vns_new;
    V_old = V_new;
    j=j+1;
end
g_d = NaN(N,Y_num);
g_b = NaN(N,Y_num);
for i=1:N
    g_d(i,:) = [Mesh_Dp(i,Pos(i,1)),Mesh_Dp(i,Pos(i,2)),Mesh_Dp(i,Pos(i,3))];
    g_b(i,:) = [Mesh_Bp(i,Pos(i,1)),Mesh_Bp(i,Pos(i,2)),Mesh_Bp(i,Pos(i,3))];
end
end