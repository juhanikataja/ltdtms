fprintf("Starting Julia daemon\n")
jlcall('', 'project', './', 'restart', true, 'threads', 'auto')
%%
fprintf("Loading LTDTMS\n")
jlcall(sprintf('()->include("%s/../../src/LTDTMS.jl")', pwd())); % TODO: add pkg with import Pkg; Pkg.add("LTDTMS")

jlcall('','project','./', 'setup', sprintf('%s/imports.jl', pwd()))

thrdata = load("inputdata.mat")
%%
fprintf('Running:\n')
tic
stats = jlcall('(T,E,K,Ethr) -> get_site_stats_ml(T,E,K,Ethr; n_samples=100, n_adapts=100)', {thrdata.MT, thrdata.EE(:,:,:), thrdata.K, thrdata.minThr})
toc
%%
statsit = jlcall('(T,E,K,Ethr) -> let statsit = LTDTMS.get_site_stats(T,E,K,Ethr;n_samples=500, n_adapts=20); Dict(:Z => [A[1] for A in statsit], :Ethr => [A[3] for A in statsit],:s =>hcat([A[2] for A in statsit]...), :d => hcat([A[4] for A in statsit]...)) end',...
  {thrdata.MT, thrdata.EE(:,:,:), thrdata.K, thrdata.minThr})

%%
