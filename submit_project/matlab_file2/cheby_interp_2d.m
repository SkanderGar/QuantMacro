function fun = cheby_interp_2d(Order,Yd,Xd,Zd)
[N,C] = size(Xd);
Y = reshape(Yd,N*C,1);
X = reshape(Xd,N*C,1);
Z = reshape(Zd,N*C,1);

min_X = min(X);
max_X = max(X);
min_Z = min(Z);
max_Z = max(Z);

Phi = cell(Order,1);
Phi2d = cell(Order,Order);
phi0 = @(x) 1;
phi1 = @(x) x;

Phi{1} = phi0;
Phi{2} = phi1;
if Order>2
    for i=3:Order
        Phi{i} = @(x) 2*x.*Phi{i-1}(x) - Phi{i-2}(x); 
    end
end

for ix=1:Order
    for jx=1:Order
        Phi2d{ix,jx} = @(x,z) Phi{ix}(x).*Phi{jx}(z);    
    end
end
X_cheb = 2*(X-min_X)/(max_X-min_X) -1;
Z_cheb = 2*(Z-min_Z)/(max_Z-min_Z) -1;
PHI = NaN(N*C,Order);
count =0;
for i=1:Order
    for j=1:Order
        count = count+1;
        PHI(:,count)=Phi2d{i,j}(X_cheb,Z_cheb);
    end
end
Theta = (PHI'*PHI)\(PHI'*Y);

fun = @(x,z) 0;
count = 0;
for it=1:Order
    for k=1:Order
        count = count+1;
        fun = @(x,z) fun(x,z) + Theta(count)*Phi2d{it,k}(2*(x-min_X)/(max_X-min_X) -1,2*(z-min_Z)/(max_Z-min_Z) -1);
    end
end
end