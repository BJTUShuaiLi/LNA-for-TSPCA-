function g = gfun(A,x)
a=tensor(A);
for i=1:(ndims(A)-1)
    a=ttv(a,x,1);    
end
g=-(ndims(A))*double(a);
end