function [u,v]=proj_grad_low_rank(X,Y,Lx,Ly,lambda,r,varargin)
    m=size(Y,1);
    n=size(X,1);
    ninj=size(Y,2);
    if ninj~=size(X,2)
        error('X and Y should have same number of columns');
    end
    if m~=size(Ly,1) | m~=size(Ly,2)
        error(sprintf('Ly should be size m=%d, square',m));
    end
    if n~=size(Lx,1) | n~=size(Lx,2)
        error(sprintf('Lx should be size n=%d, square',n));
    end
    p=inputParser;
    addOptional(p, 'Omega', NaN);
    addOptional(p, 'u0', rand(m,r)/sqrt(m*r));
    addOptional(p, 'v0', rand(n,r)/sqrt(n*r));
    addOptional(p, 'maxiter', 200);
    addOptional(p, 'step_size', 0.1);
    addOptional(p, 'tol', 1e-7);
    addOptional(p, 'sigma_armijo', 0.01);
    addOptional(p, 'beta_armijo', 0.1);
    addOptional(p, 'line_search',1);
    addOptional(p, 'momentum', 0);
    parse(p,varargin{:});
    u0=p.Results.u0;
    v0=p.Results.v0;
    tol=p.Results.tol;
    maxiter=p.Results.maxiter;
    eta=p.Results.step_size;
    sigma=p.Results.sigma_armijo;
    beta=p.Results.beta_armijo;
    Omega=p.Results.Omega;
    momentum=p.Results.momentum;
    line_search=p.Results.line_search;
    if (is_Omega(Omega) && ...
        ((ninj ~= size(Omega,2)) || (m ~= size(Omega,1))))
        error('Omega is unexpected dimension');
    end
    %% initialize and loop
    k=0; 
    u=u0; 
    v=v0;
    uold=u;
    vold=v;
    if (momentum > 0)
        uold2=u;
        vold2=v;
        umom=u;
        vmom=v;
    end
    gU=grad_U(u,v,X,Y,Lx,Ly,lambda,Omega);
    gV=grad_V(u,v,X,Y,Lx,Ly,lambda,Omega);
    init_grad_norm=norm([gU(:);gV(:)],2);
    fprintf('Initial gradient norm %f\n',init_grad_norm);
    while k < maxiter
        if (momentum > 0)
            uold2=uold;
            vold2=vold;
        end
        uold=u;
        vold=v;
        if (momentum > 0) && (k > 1)
            umom=uold+momentum*(k-2)/(k+1)*(uold-uold2);
            vmom=vold+momentum*(k-2)/(k+1)*(vold-vold2);
        end
        %fprintf('iter %d\n',k)
        k=k+1;
        %% compute gradients
        gU=grad_U(u,v,X,Y,Lx,Ly,lambda,Omega);
        gV=grad_V(u,v,X,Y,Lx,Ly,lambda,Omega);
        %% check stopping condition
        proj_grad=[gU(:); gV(:)];
        proj_grad([u(:) == 0; v(:) == 0]) = 0;
        if norm(proj_grad,2) <= tol*init_grad_norm
            fprintf('Relative tolerance reached at step %d\n', k);
            break
        end
        if (momentum > 0)
            [u,v]=proj_grad_step(umom,vmom,gU,gV,eta);
        else
            [u,v]=proj_grad_step(uold,vold,gU,gV,eta);
        end
        %% line search for step size
        if line_search==1
            if armijo_cond(u,v,uold,vold,gU,gV,X,Y,Lx,Ly,lambda,sigma,Omega)
                % take bigger step
                eta=eta/beta;
                if (momentum > 0)
                    [u,v]=proj_grad_step(umom,vmom,gU,gV,eta);
                else
                    [u,v]=proj_grad_step(uold,vold,gU,gV,eta);
                end
                while armijo_cond(u,v,uold,vold,gU,gV,X,Y,Lx,Ly,lambda, ...
                                  sigma,Omega)
                    eta=eta/beta;
                    if (momentum > 0)
                        [u,v]=proj_grad_step(umom,vmom,gU,gV,eta);
                    else
                        [u,v]=proj_grad_step(uold,vold,gU,gV,eta);
                    end
                end
                eta=eta*beta;
                if (momentum > 0)
                    [u,v]=proj_grad_step(umom,vmom,gU,gV,eta);
                else
                    [u,v]=proj_grad_step(uold,vold,gU,gV,eta);
                end
            else
                % take smaller step
                eta=eta*beta;
                if (momentum > 0)
                    [u,v]=proj_grad_step(umom,vmom,gU,gV,eta);
                else
                    [u,v]=proj_grad_step(uold,vold,gU,gV,eta);
                end
                while ~armijo_cond(u,v,uold,vold,gU,gV,X,Y,Lx,Ly,lambda, ...
                                   sigma,Omega)
                    eta=eta*beta;
                    if (momentum > 0)
                        [u,v]=proj_grad_step(umom,vmom,gU,gV,eta);
                    else
                        [u,v]=proj_grad_step(uold,vold,gU,gV,eta);
                    end
                end
            end
        end
        if f ~= 1
            fprintf('cost %d: %e (%e), eta=%e\n',...
                    k, cost(u,v,X,Y,Lx,Ly,lambda,Omega),...
                    cost_orig(u,v,X,Y,Lx,Ly,lambda,Omega),...
                    eta);
        else
            if mod(k,10) == 0
                fprintf('cost %d: %e, eta=%e\n',...
                        k,...
                        cost(u,v,X,Y,Lx,Ly,lambda,Omega),...
                        eta);
            end
        end
    end
end

function [u,v]=proj_grad_step(uold,vold,gU,gV,eta)
    u = uold - eta*gU;
    v = vold - eta*gV;
    u(u<0)=0;
    v(v<0)=0;
end

function p=innerProd(X,Y)
    x=X(:);
    y=Y(:);
    p=x'*y;
end

function t=armijo_cond(u,v,uold,vold,gU,gV,X,Y,Lx,Ly,lambda,sigma,Omega)
    grad_dot_diff=innerProd(gU,u-uold) + innerProd(gV,v-vold);
    if cost(u,v,X,Y,Lx,Ly,lambda,Omega) -...
            cost(uold,vold,X,Y,Lx,Ly,lambda,Omega) ...
            <= sigma*grad_dot_diff
        t=1;
    else
        t=0;
    end
end

function t=is_Omega(Omega)
    t=full(all(all(~isnan(Omega(:)))));
end

function Y=P_Omega(X,Omega)
    Y=X;
    Y(find(Omega)) = 0.0;
end

function c=cost_orig(U,V,X,Y,Lx,Ly,lambda,Omega)
    if ~is_Omega(Omega)
        c=norm(Y-U*V'*X,'fro')^2 + ...
          lambda*norm(U*V'*Lx' + Ly*U*V','fro')^2;
    else
        c=norm(P_Omega(Y-U*V'*X,Omega),'fro')^2 +...
          lambda*norm(U*V'*Lx' + Ly*U*V','fro')^2;
    end
end

function gU=grad_U(U,V,X,Y,Lx,Ly,lambda,Omega)
    gU=grad_U_1(U,V,X,Y,Lx,Ly,lambda,Omega);
end

function gV=grad_V(U,V,X,Y,Lx,Ly,lambda,Omega)
    gV=grad_V_1(U,V,X,Y,Lx,Ly,lambda,Omega);
end

function c=cost(U,V,X,Y,Lx,Ly,lambda,Omega)
    c=cost_1(U,V,X,Y,Lx,Ly,lambda,Omega);
end

%% original formulation
function [gU]=grad_U_1(U,V,X,Y,Lx,Ly,lambda,Omega)
    if ~is_Omega(Omega)
        gU=2*(-Y*X'*V + U*V'*X*X'*V+...
              lambda*(Ly'*U*V'*Lx' + Ly'*Ly*U*V' + ...
                      U*V'*Lx'*Lx + Ly*U*V'*Lx)*V);
    else
        gU=2*(P_Omega(U*V'*X-Y,Omega)*X'*V +...
              lambda*(Ly'*U*V'*Lx' + Ly'*Ly*U*V' + ...
                      U*V'*Lx'*Lx + Ly*U*V'*Lx)*V);
    end        
end

function [gV]=grad_V_1(U,V,X,Y,Lx,Ly,lambda,Omega)
    if ~is_Omega(Omega)
        gV=2*(-X*Y'*U + X*X'*V*U'*U +...
              lambda*(Lx*V*U'*Ly + V*U'*Ly'*Ly +...
                      Lx'*Lx*V*U' + Lx'*V*U'*Ly')*U);
    else
        gV=2*(X*P_Omega(U*V'*X-Y,Omega)'*U + ...
              lambda*(Lx*V*U'*Ly + V*U'*Ly'*Ly +...
                      Lx'*Lx*V*U' + Lx'*V*U'*Ly')*U);
        
    end
end

function c=cost_1(U,V,X,Y,Lx,Ly,lambda,Omega)
    if ~is_Omega(Omega)
        c=norm(Y-U*V'*X,'fro')^2 + ...
          lambda*norm(U*V'*Lx' + Ly*U*V','fro')^2;
    else
        c=norm(P_Omega(Y-U*V'*X,Omega),'fro')^2 +...
          lambda*norm(U*V'*Lx' + Ly*U*V','fro')^2;
    end
end

% Below are regularizations on the individual components of U, V.
% They are more efficient to evaluate, but we have not studied them fully.

%% formulation 2
function c=cost_2(U,V,X,Y,Lx,Ly,lambda,Omega)
    if ~is_Omega(Omega)
        c=norm(Y-U*V'*X,'fro')^2 + lambda*norm(Lx*V,'fro')^2 + ...
          lambda*norm(Ly*U,'fro')^2;
    else
        c=norm(P_Omega(Y-U*V'*X,Omega),'fro')^2 +...
          lambda*norm(Lx*V,'fro')^2 + ...
          lambda*norm(Ly*U,'fro')^2;
    end
end

function [gU]=grad_U_2(U,V,X,Y,Lx,Ly,lambda,Omega)
    if ~is_Omega(Omega)
        gU=2*(-Y*X'*V + U*V'*X*X'*V+...
              lambda*Ly'*Ly*U);
    else
        gU=2*(P_Omega(U*V'*X-Y,Omega)*X'*V +...
              lambda*Ly'*Ly*U);
    end
end

function [gV]=grad_V_2(U,V,X,Y,Lx,Ly,lambda,Omega)
    if ~is_Omega(Omega)
        gV=2*(-X*Y'*U + X*X'*V*U'*U +...
              lambda*Lx'*Lx*V);
    else
        gV=2*(X*P_Omega(U*V'*X-Y,Omega)'*U + ...
              lambda*Lx'*Lx*V);
    end
end

%% formulation 3, following Huang, Shen, Buja (2009)
function c=cost_3(U,V,X,Y,Lx,Ly,lambda,Omega)
    VtX=V'*X; % compute this separately to avoid computing U*V'
    if ~is_Omega(Omega)
        c=norm(Y-U*VtX,'fro')^2 + ...
          lambda*norm(Lx*V,'fro')^2*norm(U,'fro')^2 + ...
          lambda*norm(Ly*U,'fro')^2*norm(V,'fro')^2 + ...
          lambda^2*norm(Lx*V,'fro')^2*norm(Ly*U,'fro')^2;
    else
        c=norm(P_Omega(Y-U*VtX,Omega),'fro')^2 +...
          lambda*norm(Lx*V,'fro')^2*norm(U,'fro')^2 + ...
          lambda*norm(Ly*U,'fro')^2*norm(V,'fro')^2 + ...
          lambda^2*norm(Lx*V,'fro')^2*norm(Ly*U,'fro')^2;
    end
end

function [gU]=grad_U_3(U,V,X,Y,Lx,Ly,lambda,Omega)
    if ~is_Omega(Omega)
        gU=2*(-Y*X'*V + U*V'*X*X'*V + ...
              lambda*(Ly'*Ly*U*norm(V,'fro')^2 + ...
                      norm(Lx*V,'fro')^2*U + ...
                      lambda*Ly'*Ly*U*norm(Lx*V,'fro')^2 ));
    else
        gU=2*(P_Omega(U*V'*X-Y,Omega)*X'*V +...
              lambda*(Ly'*Ly*U*norm(V,'fro')^2 + ...
                      norm(Lx*V,'fro')^2*U + ...
                      lambda*Ly'*Ly*U*norm(Lx*V,'fro')^2 ));
    end
end

function [gV]=grad_V_3(U,V,X,Y,Lx,Ly,lambda,Omega)
    if ~is_Omega(Omega)
        gV=2*(-X*Y'*U + X*X'*V*U'*U + ...
              lambda*(Lx'*Lx*V*norm(U,'fro')^2 + ...
                      norm(Ly*U,'fro')^2*V + ...
                      lambda*Lx'*Lx*V*norm(Ly*U,'fro')^2 ));
    else
        gV=2*(X*P_Omega(U*V'*X-Y,Omega)'*U + ...
              lambda*(Lx'*Lx*V*norm(U,'fro')^2 + ...
                      norm(Ly*U,'fro')^2*V + ...
                      lambda*Lx'*Lx*V*norm(Ly*U,'fro')^2 ));
    end
end