function f = fun(A,x)
x=tensor(x);
x=squeeze(x);
b=x;
for i=1:(ndims(A)-1)
    b=ttt(b,x);
end
A = tensor(A);
f=-innerprod(A,b);
    
end


       