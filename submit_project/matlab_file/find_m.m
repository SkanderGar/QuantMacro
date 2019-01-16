function pos = find_m(m)
global M
[~,pos] = min(abs(M-m));
end