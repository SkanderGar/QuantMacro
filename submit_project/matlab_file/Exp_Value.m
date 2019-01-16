function E_V = Exp_Value(V,y_pos)
global n piy
Trans = piy(y_pos,:);
E_V = V*Trans';
E_V = repmat(E_V',n,n);
end
