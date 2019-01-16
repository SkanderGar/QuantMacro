function pos = find_b(b)
global B
[~,pos] = min(abs(B-b));
end