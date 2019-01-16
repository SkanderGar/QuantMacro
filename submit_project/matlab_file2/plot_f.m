f1 = figure;
subplot(1,2,1);
plot(C_grid./Y(2:end))
ylabel('c_t/y_t')
xlabel('Period')
title('Consumption to output ratio')
subplot(1,2,2);
plot(d_grid)
ylabel('d_t')
xlabel('Period')
title('Value of default')
%%
f2 = figure;
plot(B_grid)
ylabel('b_t')
xlabel('Period')
title('Debt')
%%
g_d_mat = cell(3,1);
g_b_mat = cell(3,1);
q_mat = cell(3,1);
Bi = cell(3,1);
Di = cell(3,1);
for i =1:3
g_d_mat{i} = reshape(g_d(:,1),n,n);
g_b_mat{i} = reshape(g_b(:,1),n,n);
q_mat{i} = reshape(q(:,1),n,n);
Bi = reshape(Mesh_B(:,1),n,n);
Di = reshape(Mesh_D(:,1),n,n);
end
f3 = figure;
subplot(1,2,1);
plot3(Bi,Di,g_d_mat{1},'b')
hold on
plot3(Bi,Di,g_d_mat{2},'r')
hold on
plot3(Bi,Di,g_d_mat{3},'k')
zlabel('d_{t+1}')
ylabel('d_t')
xlabel('b_t')
legend('y = 1','y = 1.5','y = 2')
title('Policy for the default share')
subplot(1,2,2);
plot3(Bi,Di,g_b_mat{3},'k')
zlabel('b_{t+1}')
ylabel('d_t')
xlabel('b_t')
title('Policy for bonds')

%%
f4 = figure;
plot3(Bi,Di,q_mat{3},'k')
zlabel('q_{t+1}')
ylabel('d_t')
xlabel('b_t')
title('Bonds Prices')

%basevalue = -2;
%area([1995 1996],[2.5 2.5],basevalue,'FaceColor','b','FaceAlpha',.3,'EdgeAlpha',.3)
%area([1998 1999],[2.5 2.5],basevalue,'FaceColor','b','FaceAlpha',.3,'EdgeAlpha',.3)
%area([2004 2005],[2.5 2.5],basevalue,'FaceColor','b','FaceAlpha',.3,'EdgeAlpha',.3)
%area([2010 2011],[2.5 2.5],basevalue,'FaceColor','b','FaceAlpha',.3,'EdgeAlpha',.3)
%area([2016 2017],[2.5 2.5],basevalue,'FaceColor','b','FaceAlpha',.3,'EdgeAlpha',.3)