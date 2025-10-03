% Requires chunkIE: https://github.com/fastalgorithms/chunkie
% and chebfun: https://www.chebfun.org/
n_vals = [6 10 14 20 ]; % Order of Gauss panels
nref_vals = [ 1 2 3 4 5 ];  % Number of levels of uniform mesh refinement
rect = [-1 1 -1 1]; % Bounds of the computational domain
zk_vals = [20 100]; % Wavenumber. Inverse scattering example uses 20, forward example uses 100.

mkdir('../data/examples/SD_matrices');

for zk=zk_vals
    disp(['Computing matrices for wavenumber zk = ' num2str(zk)])
    
    % Loop over order of the Gauss panels
    for i=1:length(n_vals)
        n = n_vals(i);
        % Loop over numbers of levels of uniform mesh refinement
        for j=1:length(nref_vals)
            nref = nref_vals(j);
            chnkr = squarechunker(n, nref, rect);
            Skern = kernel('helmholtz', 's', zk);
            Dkern = kernel('helmholtz', 'd', zk);
            S = chunkermat(chnkr, Skern);
            D = chunkermat(chnkr, Dkern);
            filename = sprintf('../data/examples/SD_matrices/SD_k%d_n%d_nside%d_dom%g.mat', zk, n, 2^nref, rect(2));
            
            % print update that computation has finished
            disp(['Saving to ' filename])
            
            save(filename, 'S', 'D', '-v7.3')
            % print update that saving has finished
            disp('Done saving file')
        end
    end
end