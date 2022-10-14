fprintf("Starting Julia daemon\n")
jlcall('', 'project', './', 'restart', true, 'threads', 'auto')
%%
fprintf("Loading LTDTMS\n")
% jlcall(sprintf('()->include("%s/../../src/LTDTMS.jl")', pwd())); 
jlcall('import LTDTMS');

jlcall('','project','./', 'setup', sprintf('%s/imports.jl', pwd()))

thrdata = load("inputdata.mat")
%%
fprintf('Running:\n')
stats = jlcall('(T,E,K,Ethr) -> get_site_stats_ml(T,E,K,Ethr; n_samples=100, n_adapts=100)', {thrdata.MT, thrdata.EE(:,:,:), thrdata.K, thrdata.minThr})
