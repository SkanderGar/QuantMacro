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
f3 = figure;
subplot(1,2,1);
plot(B',g_d(:,1),'b')
hold on
plot(B',g_d(:,2),'r')
hold on
plot(B',g_d(:,3),'k')
ylabel('d_t')
xlabel('b_t')
legend('y = 1','y = 1.5','y = 2')
title('Policy for the default share')
subplot(1,2,2);
plot(B',g_b(:,1),'b')
hold on
plot(B',g_b(:,2),'r')
hold on
plot(B',g_b(:,3),'k')
ylabel('b_{t+1}')
xlabel('b_t')
legend('y = 1','y = 1.5','y = 2')
title('Policy for bonds')

%%
f4 = figure;
plot(B',q(:,1),'b')
hold on
plot(B',q(:,2),'r')
hold on
plot(B',q(:,3),'k')
ylabel('Price')
xlabel('Quantity')
legend('y = 1','y = 1.5','y = 2')
title('Bonds Prices')

%basevalue = -2;
%area([1995 1996],[2.5 2.5],basevalue,'FaceColor','b','FaceAlpha',.3,'EdgeAlpha',.3)
%area([1998 1999],[2.5 2.5],basevalue,'FaceColor','b','FaceAlpha',.3,'EdgeAlpha',.3)
%area([2004 2005],[2.5 2.5],basevalue,'FaceColor','b','FaceAlpha',.3,'EdgeAlpha',.3)
%area([2010 2011],[2.5 2.5],basevalue,'FaceColor','b','FaceAlpha',.3,'EdgeAlpha',.3)
%area([2016 2017],[2.5 2.5],basevalue,'FaceColor','b','FaceAlpha',.3,'EdgeAlpha',.3)