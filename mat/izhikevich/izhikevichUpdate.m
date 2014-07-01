function [v, u, firing] = izhikevichUpdate(v,u,I_syn,a,b,c,d,id)

 if id==1               %Thalamic input
    I=I_syn+5*randn(); 
 else
    I=I_syn+2*randn();
 end
 
 v=v+0.5*(0.04*v^2+5*v+140-u+I);  %Two steps of 0.5 ms for numerical stability
 v=v+0.5*(0.04*v^2+5*v+140-u+I);
 u=u+1*a*(b*v-u);
 
 if v>=30
    v=c;
    firing=1;
    u=u+d;
 else
 firing=0;
 end

end