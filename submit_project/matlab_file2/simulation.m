function [Y,B_grid,d_grid,D_grid,q_grid,C_grid] = simulation(T,g_d,g_b,D)
global q y_grid piy lamda tau omega1 omega2

function idx = find_p(d,b)
    global Mesh_D Mesh_B
    D_mat_f = Mesh_D(:,1);
    b_mat_f = Mesh_B(:,1);
    min_d = min(abs(d-D_mat_f));
    idx_d = find(abs(d-D_mat_f)==min_d);
    min_b = min(abs(b-b_mat_f));
    idx_b = find(abs(b-b_mat_f)==min_b);
    idx = intersect(idx_d,idx_b);
end

[~,c]=size(piy);
y0_idx=1;

b0 = 0;
q0=1;
d0 = 0;

idx = find_p(d0,b0);

Y_pos = NaN(T+1,1);
Y = NaN(T+1,1);
B_grid = NaN(T+1,1);
d_grid = NaN(T+1,1);
D_grid = NaN(T+1,1);
q_grid = NaN(T+1,1);
C_grid = NaN(T,1);


Y_pos(1) = y0_idx;
Y(1)= y_grid(y0_idx);
B_grid(1)= b0;
q_grid(1)= q0;
d_grid(1) = d0;
D_grid(1) = 0;

U = rand(T,1);
for i=1:T
    Pi = piy(y0_idx,:);
    C_Pi = cumsum(Pi);
    C_Pi = [0,C_Pi]; 
    for j=2:c+1
        if C_Pi(j-1)<=U(i,1)&&U(i,1)<C_Pi(j)
             y1_idx=j-1;
        end
    end
    if U(i,1) == 1
        y1_idx=c;
    end
    Y_pos(i+1) = y1_idx;
    Y(i+1) = y_grid(y1_idx);
    
    if D_grid(i)==0
        bp = g_b(idx,Y_pos(i));
        dp = g_d(idx,Y_pos(i));
        d_grid(i+1) = dp;
        B_grid(i+1)=bp;
        idx_1 = find_p(dp,bp); 
        q_grid(i+1)= q(idx_1,Y_pos(i));
        D_grid(i+1) = D(idx_1,Y_pos(i)); 
        C_grid(i) = Y(i) + q_grid(i+1)*B_grid(i+1)-B_grid(i);
    elseif D_grid(i)==1
        dp = g_d(idx,Y_pos(i));
        bp = B_grid(i)*(1-dp);
        d_grid(i+1) = dp;
        B_grid(i+1)=bp;
        idx_1 = find_p(dp,bp);
        q_grid(i+1)= q(idx_1,Y_pos(i));
        
        F=omega1*d_grid(i+1) + omega2*d_grid(i);
        prob = lamda*F;
        ui = rand(1);
        if ui<=prob
           D_grid(i+1)=1;
           d_grid(i+1) = g_d(idx_1,Y_pos(i));
           C_grid(i) = Y(i)*(1-tau(Y_pos(i))*F);
        elseif ui>prob
            D_grid(i+1)=0;
            d_grid(i+1) = 0;
            C_grid(i) = Y(i)*(1-tau(Y_pos(i))*F);
        end
    end
    idx = idx_1;
end
end