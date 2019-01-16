function u = U(c)
global sig
[N,S] = size(c);
u = -inf(N,S);
idx = c>0;
u(idx) = c(idx).^(1-sig) ./(1-sig);
end