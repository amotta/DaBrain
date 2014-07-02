% Simulation on GPU
clear all

mydev=gpuDevice();
parallel.gpu.rng('shuffle','Philox4x32-10')

f = 0.005;

% Excitatory neurons            Inhibitory neurons
Ne = round(800 * f);            Ni = round(200*f);                      %Number of different neurons
re=gpuArray.rand(Ne,1);          ri=gpuArray.rand(Ni,1);        %Random number to differentiate neurons

%Values for the differnt neurons
a=[0.02*gpuArray.ones(Ne,1);    0.02+0.08*ri];                  %Neuronal parameter
b=[0.2*gpuArray.ones(Ne,1);     0.25-0.05*ri];                  %Neuronal parameter
c=[-65+15*re.^2;                -65*gpuArray.ones(Ni,1)];       %Membrane voltage reset
d=[8-6*re.^2;                   2*gpuArray.ones(Ni,1)];         %Recovery variable reset
S=[0.5*gpuArray.rand(Ne+Ni,Ne),  -gpuArray.rand(Ne+Ni,Ni)];     %"Synpase matrix"
id=[gpuArray.ones(Ne,1);        2*gpuArray.ones(Ni,1);];        % 1=excitatory 2=inhibitory

%Init values
v=-65*gpuArray.ones(Ne+Ni,1);           % Initial values of v
u=b.*v;                                 % Initial values of u
firing=gpuArray.zeros(Ne+Ni,1);     

fired=zeros(1000,Ne+Ni);                % Pre-allocated for saving data

tic
for t=1:1000           % simulation of 1000 ms
    
  I_syn=S*firing;
  [v, u, firing]=arrayfun(@izhikevichUpdate,v,u,I_syn,a,b,c,d,id); 
  fired(t,:)=gather(firing);
  
end
toc

[fire_time, fire_id]=find(fired);
plot(fire_time, fire_id,'.')



