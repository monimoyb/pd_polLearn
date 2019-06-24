%% Function to append state matrices over horizon 

function [Ax_vec, Bx_vec] = appendMat(N_mpc, nx, nu, A, B)

Ax_vec = nan(N_mpc*nx, nx);
        for ii = 1 : N_mpc
            Ax_vec(1+(ii-1)*nx:ii*nx,:) = A^ii;
        end

Bx_vec = zeros(N_mpc*nx, nu*N_mpc);

        for ii = 0 : N_mpc-1
            for jj = 0 : ii-1
                Bx_vec(1+ii*nx:(ii+1)*nx, 1+jj*nu:  (jj+1)*nu) = A^(ii-jj)*B;
            end
            Bx_vec(1+ii*nx:(ii+1)*nx, 1+ii*nu:(ii+1)*nu) = B;
        end

end