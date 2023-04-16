function he = hefun(A,x)
a=tensor(A);
for i=1:(ndims(A)-2)
    a=ttv(a,x,1); 
end
he=-ndims(A)*(ndims(A)-1)*double(a);
end