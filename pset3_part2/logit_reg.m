cd('C:\Users\DELL\Desktop\Luis\Pset3')
[data, data_txt] = xlsread('data_def.xls');
Y = data(:,3);
idx = data(:,1);
X = mat_ind(idx);
X = [X, data(:,4)];
% Logit regression. Binomial outcome (0, 1) with fixed effect
[par_est,dev,stats]=glmfit(X,Y,'binomial','constant','off');

%9.3402 the data gives us this estimate for the beta