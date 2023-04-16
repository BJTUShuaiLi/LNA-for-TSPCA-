function NLf = NLfun(x,y)
global A n
NLF = hefun(x)-2*y*eye(length(x));
end