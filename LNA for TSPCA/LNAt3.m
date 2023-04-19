function [xopt,Out] = LNAt3(n, s, data,  probname, pars)

% This code aims at solving the tensor sparse PCA with form
%
%         min_{x\in R^n} -\cal{A}x^3  s.t. x^{T}x=1, \|x\|_0<=s
%
% where s is the given sparsity, which is << n.
%
% Inputs:
%     n       : Dimension of the solution x, (required)
%     s       : Sparsity level of x, an integer between 1 and n-1, (required)
%     data    : 
%   
%               data.A  --  3 order n dims hyper symmetric tensor (required)
%     probname: Name of problem, should be {'TSPCA'}
%     pars:     Parameters are all OPTIONAL
%               pars.x0     --  Starting point of x,   pars.x0=zeros(n,1) (default)
%               pars.y0     --  Starting point of y,   pars.y0=zeros(m,1) (default)
%               pars.beta   --  A positive parameter,  a default one is given related to inputs
%               pars.IterOn --  Results will  be shown for each iteration if pars.IterOn=1 (default)
%                               Results won't be shown for each iteration if pars.IterOn=0
%
% Outputs:
%     Out.sol:           The sparse solution x
%     Out.sparsity:      Sparsity level of Out.sol
%     Out.error:         Error used to terminate this solver
%     Out.time           CPU time
%     Out.iter:          Number of iterations
%     Out.obj:           Objective function value at Out.sol
%
%

warning off;
t0     = tic;

if  nargin<3         
    disp(' No enough inputs. No problems will be solved!'); return;   
elseif nargin==3
    disp(' We will solve this problem in a general way');
    probname = 'general_example';  
end

if nargin>=3
    if nargin<5; pars=[]; end
    if isfield(pars,'IterOn');iterOn = pars.IterOn; else; iterOn = 1;     end     
    if isfield(pars,'MaxIt'); ItMax  = pars.MaxIt;  else; ItMax  = 2000;  end
    if isfield(pars,'x0');    x0     = pars.x0;     else; x0 = zeros(n,1);end
    if isfield(pars,'y0');    y0     = pars.y0;     else; y0 = zeros(m,1);end
    if isfield(pars,'beta');  beta  = pars.beta;    else; beta   = 0.5;  end
end

x       = x0;
y       = y0;
tol     = 1e-6;
err     = 0;
obj     = 0;
f       = [];
fprintf(' Start to run the sover...\n');
if iterOn
    fprintf('\n Iter             Error                Ojective \n');      
    fprintf('--------------------------------------------------------\n');
end

% The main body

switch      probname
    case 'TSPCA'; [xopt,iter] = LNAt3_TSPCA(x,y,data);
%     otherwise  ; 
end

xopt(abs(xopt)<1e-4)=0;

% results output
time        = toc(t0);
Out.sparsity= nnz(xopt); 
Out.error   = err;
Out.time    = time;
Out.iter    = iter;
Out.obj     = obj;
Out.f = f;

if iterOn
    fprintf(' --------------------------------\n');
    fprintf(' Obj :   %5.2f\n', Out.obj );
    fprintf(' Time:  %5.2f second\n', Out.time);
    fprintf(' Iter: %4d  \n\n', Out.iter);
end


%fprintf('\n--------------------------------------------------------\n');
function [x,iter] = LNAt3_TSPCA(x,y,data)
A = data.A;
%if isfield(pars,'scale'); scale  = pars.scale;  else; scale =  svds(A,1);end     
if isfield(pars,'scale'); scale  = pars.scale;  else; scale =  max(max(max(A)));end     
if scale > 1; A = A/scale; end
gradL  = gfun(A,x)-2*y*x;
[~,T]  = maxk(abs(x-beta*gradL),s);
obj    = fun(A(T,T,T), x(T));

if iterOn
    fprintf(' Iter       Error       Objective\n');
    fprintf(' --------------------------------\n');
end

inner = 0;
for iter  = 1:ItMax
%---------------------------------find Tk----------------------------------
gradL  = gfun(A(T,T,:),x(T))-2*y*x;     
[~,T]  = maxk(abs(x - beta*gradL),s);
Tc     = setdiff(1:n,T);    
%----------------------------- Stop criterion -----------------------------
ereq   = sum(x(T).*x(T))-1;
err    = norm(gradL(T))+ norm(x(Tc))+ abs(ereq)+...
        max(0,max(abs(gradL(Tc)))-min(abs(x(T)))/beta);
%if iterOn
    %objt = obj;
    %fprintf('%4d       %5.2e      %5.4f\n',iter,err,objt);
%end
if iterOn
    if scale>1; objt = scale*obj;
    else
        objt = obj;
    end
    fprintf('%4d       %5.2e      %5.4f\n',iter,err,objt);
end

if err<=tol; break;  end

%---------------------------- Newton Step----------------------------------
he = hefun(A,x);
g = gfun(A(T,T,:),x(T));
ATT    = -he(T,T)+2*y*speye(s);
d      = -[ATT 2*x(T); 2*x(T)' 0 ]\[(-g(T)+2*y*x(T));ereq];  

if max(isnan(d))==1  
    d      = -([ATT 2*x(T); 2*x(T)' 0 ]+1e-6*speye(s+1))\[(-g(T)+2*y*x(T));ereq];
end

x0     = x;
obj0   = obj;

x(T)   = x0(T) + d(1:s);


obj    = fun(A(T,T,T), x(T));
f = [f,obj];

desd   = 1e-5* sum(d(1:s).^2);
alpha  = 1;
for i  = 1:inner % inner=0 full newton inner>0 line search
    if obj < obj0 -  alpha*desd; break; end
    alpha = alpha*0.5;
    x(T)  = x0(T) + alpha*d(1:s);
    obj   = fun(A(T,T), x(T));
end

x(Tc)  = 0;


y      = y + alpha*d(s+1);

if mod(iter,20)==0   
    %beta   = norm(x)/norm(-AyE(:,T)*x(T));
    beta   = max(1e-4,beta*0.8);
end

end
if scale>1; obj=scale*obj; end


end


end
