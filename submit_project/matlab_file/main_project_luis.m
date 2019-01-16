betta=0.95;
y_grid=[0.6, 1, 1.5];
piy=[0.5, 0.3, 0.2;...
     0.05, 0.65, 0.3;...
     0.02, 0.55, 0.43];
sig=1.5;
u = @(c) c.^(1-sig)./(1-sig);
lamda=0.5;
tau=[0.1,0.4,0.5];
B=0:0.05:2.8;
max_iter=1000;
max_iterq=1000;
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

figure
plot_q(B,q)
%% Simulation of the model
% recovering the policy function

for iy=1:size(y_grid,2)
    for ib=1:size(B,2)
       % No default value and debt issuance (conditional on no deault)
        [V_ND(ib,iy),bp(ib,iy)]=max(u(max(y_grid(iy)+q(:,iy).*B'-B(ib),0))+betta*V*piy(iy,:)')  ; 
        % Default decision
        Dp(ib,iy)=(V_ND(ib,iy)<=Vd(iy)) ;   
    end
end
%% 
% Simuated sequence of GDP
% 
%  starting value (index)

yt=1;

for t=2:500
    draw_t=rand;
    yt(t)=1+(draw_t>=piy(yt(t-1),1))+(draw_t>=sum(piy(yt(t-1),1:2)));
end

%% 
% initial level of debt
% 
% index in B

bt=ones(500,1);
%% 
% index for the default decision =1 and the default state

Def_b=nan(1,500);
Def_state(500)=0;

for t=2:500
    if Def_state(t-1)==0
      %default decision (decided at t)
        Def_b(t)=Dp(bt(t-1),yt(t));
        if Def_b(t)==0
      %Debt issuance decision (decided at t)  
            bt(t)=bp(bt(t-1),yt(t));
            Def_state(t)=0;
        else
            bt(t)=1;
            Def_state(t)=1;
        end
    elseif rand<=lamda
        Def_b(t)=Dp(bt(t-1),yt(t));
        if Def_b(t)==0
        %Debt issuance decision (decided at t)   
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
%0bserved risk spread (1/q-1)
r_spread(t)=1/q(bt(t),yt(t))-1;

%Default probability
p_model(t)=1-q(bt(t),yt(t));
   
end
       
%% 
%  graph of default and the risk premia

risk_premia_graph(Def_state, r_spread)
%% Estimating a logit
N=sum(1-(isnan(Def_b)));
X=[B(bt)'./y_grid(yt)'];
Y=Def_b';
[par_est,dev,stats]=glmfit(X(1:end-1),Y(2:end),'binomial');
% Estimated Default probability
X_grid=0:0.1:2;
p_est=1./(1+exp(-par_est(1)-par_est(2)*X_grid));
Fit_model_graph(X(1:end-1), Y(2:end), X_grid, p_est)
% default probability in the simulation
p_est_sim=1./(1+exp(-par_est(1)-par_est(2)*X));
sim_and_model_graph([p_est_sim,p_model'])