function X = mat_ind(idx)
[n,c] = size(idx);
j=1;
count=0;
X = ones(n,1);
length = 0;
for ix=2:n
    count=count+1;
    if idx(ix-1) ~= idx(ix)
        x = [zeros(1,length), ones(1,count),zeros(1,n-count-length)]';
        X = [X,x];
        j=j+1;
        length = length + count;
        count=0;
    end
end
x = [zeros(1,length), ones(1,n-length)]';
X = [X,x];
X = X(:,2:end);
end
