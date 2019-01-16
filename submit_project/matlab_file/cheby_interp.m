function fun = cheby_interp(Order,Y,X)
N = length(X);
min_X = min(X);
max_X = max(X);
Phi = cell(Order,1);
phi0 = @(x) 1;
phi1 = @(x) x;

Phi{1} = phi0;
Phi{2} = phi1;
if Order>2
    for i=3:Order
        Phi{i} = @(x) 2*x.*Phi{i-1}(x) - Phi{i-2}(x); 
    end
end
X_cheb = 2*(X-min_X)/(max_X-min_X) -1;
PHI = NaN(N,Order);
for j=1:Order
    PHI(:,j)=Phi{j}(X_cheb);
end
Theta = (PHI'*PHI)\(PHI'*Y);

fun = @(X) Theta(1)*Phi{1}(2*(X-min_X)/(max_X-min_X) -1);
for k=2:Order
    fun = @(X) fun(X) + Theta(k)*Phi{k}(2*(X-min_X)/(max_X-min_X) -1);
end
end