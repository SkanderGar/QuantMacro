

betta=0.95;

y_grid=[0.6, 1, 1.5];

piy=[0.5, 0.3, 0.2;...
     0.05, 0.65, 0.3;...
     0.02, 0.55, 0.43];

sig=1.5;
u = @(c) c.^(1-sig)./(1-sig);
R=1;
lamda=0.5;
tau=[0.1,0.4,0.5];
%tau = [0, 0.5, 0.1];
B=0:0.05:2.8;

max_iter=1000;
max_iterq=1000;

% keep the same output for all the simulation
yt=1;
for t=2:500
    draw_t=rand;
    yt(t)=1+(draw_t>=piy(yt(t-1),1))+(draw_t>=sum(piy(yt(t-1),1:2)));
end

err = 1;
Tol = 0.01;
tau_lb = [0,0,0];
tau_ub = [1,1,1];
beta_lb = 0.95;
beta_ub = 1;
eta = [0.1,0.1,0.1];
eta_b = 0.1;
target = 9.3402;
j = 0;

while err>Tol
V= ones(size(B,2),1)*u(y_grid)/(1-betta);
Vd = u((1-tau).*y_grid)/(1-betta);
q=ones(size(B,2),size(y_grid,2));
for iter_q=1:max_iterq
    for iter=1:max_iter
        for iy=1:size(y_grid,2)
            Vd(iy)=u((1-tau(iy))*y_grid(iy))+(1-lamda)*betta*piy(iy,:)*Vd'+lamda*betta*V(1,:)*piy(iy,:)';
            for ib=1:size(B,2)
                 V_old=V;
                 Vd_old=Vd; 
                 V(ib,iy)=max(max(u(max(y_grid(iy)+q(:,iy).*B'-B(ib),0))+betta*V*piy(iy,:)'),Vd(iy)) ;
            end
        end
        dev= max(max(abs([V_old-V;Vd_old-Vd])));
        if dev<=0.000000000001
            break
        end
    end
    for iy=1:size(y_grid,2)
        for ib=1:size(B,2)
            q(ib,iy)=1-piy(iy,:)*(V(ib,:)<=Vd)';
        end
    end
end
%% Bond price menu
%figure
%plot_q(B,q)
%% Simulation of the model
 for iy=1:size(y_grid,2)
     for ib=1:size(B,2)
         [V_ND(ib,iy),bp(ib,iy)]=max(u(max(y_grid(iy)+q(:,iy).*B'-B(ib),0))+betta*V*piy(iy,:)')  ; 
         Dp(ib,iy)=(V_ND(ib,iy)<=Vd(iy)) ;   
     end
 end

bt=ones(500,1);
Def_b=nan(1,500);
Def_state(500)=0;

for t=2:500
    if Def_state(t-1)==0
        Def_b(t)=Dp(bt(t-1),yt(t));
        if Def_b(t)==0
            bt(t)=bp(bt(t-1),yt(t));
            Def_state(t)=0;
        else
            bt(t)=1;
            Def_state(t)=1;
        end
    elseif rand<=lamda
        Def_b(t)=Dp(bt(t-1),yt(t));
        if Def_b(t)==0
            bt(t)=bp(bt(t-1),yt(t));
            Def_state(t)=0;
        else
            bt(t)=1;
            Def_state(t)=1;
        end
    else
        Def_state(t)=1;
        bt(t)=1;
    end
    r_spread(t)=1/q(bt(t),yt(t))-1;
    p_model(t)=1-q(bt(t),yt(t));   
end

X=[B(bt)'./y_grid(yt)'];
Y=Def_b';
[par_est,dev,stats]=glmfit(X(1:end-1),Y(2:end),'binomial');
beta_comp = par_est(end);
err = beta_comp-target;

if err<0
    tau_ub = tau;
    if betta ~= 0.95
        beta_ub = betta;
    end
    tau = (1-eta).*tau + eta.*tau_lb;
    betta = (1-eta_b)*betta+ eta_b*beta_lb;
elseif err>=0
    tau_lb = tau;
    if betta ~= 0.95
        beta_lb = betta;
    end
    tau = (1-eta).*tau+ eta.*tau_ub;
    betta = (1-eta_b)*betta+ eta_b*beta_ub;
end
j=j+1;
fprintf('Iteration: %i\n',j);
fprintf('Error: %f\n',err);
fprintf('Logit Simulation: %f\n',beta_comp);
fprintf('tau: %f\n',tau);
fprintf('beta: %f\n',betta);
if j>20
    break
end
err = abs(err);
end
       
