function C = nneg_lasso( y, S, lambda, a, iter)
%Lasso spectral unmixing with non-negativity constraint
%   This function performs lasso spectral unmixing on hyperspectral images,
%   L1 norm is added as a pixel-wise constraint constraint. Non negativity
%   is added on the concentration maps using the ADMM method.
%   Input:
%           y       --> 3D hyperspectral image stack
%           S       --> Spectral profiles for pure components
%           lambda  --> Sparsity level for the imgaes
%           a       --> ADMM parameter controlling convergence speed,
%                       default is 1
%           iter    --> number of iterations for ADMM update
%   Output:
%           C       --> Concentration maps

%% Initialize parameters
y           = double(y);
[Nx,Ny,Nz]  = size(y);            % 3 dimensions from the input data
N           = Nx*Ny*Nz;
[~,k]       = size (S);           % k is number of pure components
C           = zeros(Nx, Ny,k);    % main variable C
u           = zeros(Nx, Ny,k);    % ADMM variable u
vhat        = zeros(Nx, Ny,k);    % ADMM variable vhat
R_positive  = zeros(Nx, Ny,k);    % Data space for non negativity

%% Begin iteration

fprintf('Iter \t residualC \t residualv \t residualu \n');
for ii = 1:iter
    C_old    = C;
    vhat_old = vhat;
    u_old    = u;
    
    
    % Update c
    ctilde = vhat - u;
    S_tilde = [S;eye(k)];
    parfor i = 1:Nx
        for j = 1:Ny
            if ii ==1
                y_sp = reshape(y(i,j,:),[Nz,1]);
                rhs = [y_sp; sqrt(a)*reshape(ctilde(Nx,Ny,:),[k,1])];
                c_single_pixel = lasso(S_tilde,rhs,'lambda',lambda,...
                    'MaxIter',1e5, 'Alpha', 1);
                C(i,j,:) = reshape(c_single_pixel,[1,1,k]);
            elseif min(C_old(i,j,:))<0
                y_sp = reshape(y(i,j,:),[Nz,1]);
                rhs = [y_sp; sqrt(a)*reshape(ctilde(Nx,Ny,:),[k,1])];
                c_single_pixel = lasso(S_tilde,rhs,'lambda',lambda,...
                    'MaxIter',1e5, 'Alpha', 1);
                C(i,j,:) = reshape(c_single_pixel,[1,1,k]);
            end
        end
    end
    
    % Update vhat
    vhat = max((C+u),R_positive);
    
    %Update u
    u    = u + (C-vhat);
    
    %calculate residual
    residualC    = (1/sqrt(N))*(sqrt(sum(sum(sum((C-C_old).^2)))));
    residualvhat = (1/sqrt(N))*(sqrt(sum(sum(sum((vhat-vhat_old).^2)))));
    residualu    = (1/sqrt(N))*(sqrt(sum(sum(sum((u-u_old).^2)))));
    
    fprintf('%3g \t %3.5e \t %3.5e \t %3.5e \n', ii, residualC, residualvhat, residualu);
    
end

end