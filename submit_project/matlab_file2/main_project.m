global n N omega1 omega2 sig piy Order Y_num Mesh_Bp Mesh_B Mesh_D Mesh_Dp q y_grid betta lamda B tau Dg
%parameters
eta = 0.1;
R=1;
omega1 = 1/2;
omega2 = 1/2;
Order = 3;
Y_num = 3; % number of states
n=20;
N = n^2;
betta=0.95;
y_grid=[1, 1.5, 2];
piy=[0.5, 0.3, 0.2;...
     0.05, 0.65, 0.3;...
     0.02, 0.55, 0.43];
sig=1.5;
lamda=0.5;
tau=[0.1,0.4,0.6];
%bounds
max_b = 1;
max_d = 1;
min_b = 0;
min_d = 0;
%tools
o = ones(1,n);
%sets
B=linspace(min_b,max_b,n);
D=linspace(min_d,max_d,n);
Dg = D;
%meshs
Mesh_B = repmat(B',n,N);
Mesh_Bp = Mesh_B';

Mesh_D = kron(D,o);
Mesh_D = repmat(Mesh_D',1,N);
Mesh_Dp = Mesh_D';
% prep V0
Vs = ones(N,1)*U(y_grid)/(1-betta);
Vns = ones(N,1)*U(y_grid)/(1-betta);
%prep q0 pi0
q = ones(N,3);
%function help
q_update = NaN(N,3);

err = 1;
Tol = 10^(-3);
Tol2 = 10^(-3);
j = 1;
while err>Tol
    [D,~,Vs_new,Vns_new,~,~,g_d] = VFI(Tol2, Vs, Vns);
    Vs = Vs_new;
    Vns = Vns_new;
    
    for jx=1:Y_num
        Tr = piy(jx,:);
        q_update(:,jx) = (1 - D(:,1).*g_d(:,1))*Tr(1)/R;
        for kx=2:Y_num
            q_update(:,jx) = q_update(:,jx) + (1 - D(:,kx).*g_d(:,kx))*Tr(kx)/R;
        end
    end
    %%%update q
    for lx=1:Y_num
        q(:,lx) = (1-eta)*q(:,lx)+eta*q_update(:,lx);
    end
    err = max(max(abs(q-q_update)));
    fprintf('Iteration Prices: %d \n',j);
    fprintf('Error: %d \n',err);
    j=j+1;
end
[D,V_new,Vs_new,Vns_new,Pos,g_b,g_d] = VFI(Tol, Vs, Vns);
%%
%policy for simulations
T=1000;
[Y,B_grid,d_grid,D_grid,q_grid,C_grid] = simulation(T,g_d,g_b,D);
