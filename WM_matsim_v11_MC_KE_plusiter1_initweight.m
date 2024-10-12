clear all
close all
rand('seed',0)

abr=0;

tic
M = 3000;%number of MC simulations 
%% Inlet stream 
f11 = 800;
xin(1) = 9.67/100; %PET
xin(2) = 0/100;
xin(3) = 0/100;
xin(4) = 3.61/100; %LDPE
xin(5) = 0/100;
xin(6) = 2.25/100; %HDPE
xin(7) = 0.15/100; %Haz
xin(8) = 2.71/100; %Fe
xin(9) = 0.94/100; %Al
xin(10) = 44.51/100; %RDF
xin(11) = 13.33/100; %Card
xin(12) = 22.83/100; %Paper


%% Definition of technology
types = [2; 3; 4; 6; 5; 8; 11]; 
Nber=length(types); %number of units
types(8:14)=0; %env. types 
%1 zsáktépõ 
%2 aprító 
%3 mágneses 
%4 alu 
%5 ballisztikus 
%6 dobrosta (80) 
%7 NIR 
%8 HDPE 
%9 PET víztiszta 
%10 PET tarka 
%11 fólia optika 
%12 fólia tiszta 
%13 TX optika 
%14 légszeparátor 
%15 dobrosta (60) 
%16 dobrosta (20) %lehet még kell

edges=[1 2; 
       2 3;
       3 4;
       4 5;
       5 6;
       5 7; 
       2 8;
       3 9;
       4 10;
       6 11;
       6 12;
       7 13;
       7 14];  
G = digraph(edges(:,1),edges(:,2));

if abr==1
figure(1); plot(G)
end
A=full(adjacency(G));
Ns=size(A,1); %number of states 
u=zeros(Ns,1); %structure of the input vector, the first element is updated
%% Get the separation paramaters
%load('ratios.mat')
%x=[2;2;3;3;4;4;6;6;5;5;8;8;11;11]; %units in the technology
B1 = zeros(14,12);
B2 = B1;

A1 = [100	100	95	99	97	99	95	97	83	95	5	10	91	96
      100	100	93	95	97	99	94	96	4	8	82	92	2	5
      100	100	95	99	97	99	94	98	92	97	95	98	92	98
      100	100	98	99	98	99	98	99	98	99	98	99	98	99
      100	100	5	15	98	99	95	96	96	98	97	99	90	99
      100	100	95	99	2	10	83	85	40	60	96	99	88	99
      100	100	95	99	97	99	20	40	40	60	88	94	88	99
      100	100	97	99	97	99	90	95	10	30	90	95	90	95
      100	100	95	99	97	99	65	80	15	35	85	90	90	95]'; %separation efficiencies by Exp1 (min and max, by components)
B1(:,1) = A1(:,1);
B1(:,4) = A1(:,2);
B1(:,6:end) = A1(:,3:end); %because xin(2;3;5)=0 
ratios1 = B1;

A2 = [100	100	98	99	98	99	99	100	98	99	5	6	97	98
      100	100	94	95	97	98	99	100	4	5	91	93	10	15
      100	100	99	100	98	99	99	100	98	99	94	95	97	98
      100	100	98	99	98	99	95	98	98	99	93	95	93	95
      100	100	5	8	99	100	98	99	98	99	98	99	98	99
      100	100	99	100	2	3	98	99	1	2	98	99	98	99
      100	100	94	95	98	99	60	70	1	2	90	95	90	95
      100	100	98	99	98	99	98	99	5	10	92	95	92	95
      100	100	98	99	97	98	85	90	5	10	92	95	90	92]'; %separation efficiencies by Exp1
B2(:,1) = A2(:,1);
B2(:,4) = A2(:,2);
B2(:,6:end) = A2(:,3:end);
ratios2 = B2;

%% Aggregated distribution of Experts for separation parameters
R_min1 = ratios1(1:2:end,:)';
R_min2 = ratios2(1:2:end,:)';
R_max1 = ratios1(2:2:end,:)';
R_max2 = ratios2(2:2:end,:)';

%ratios by experts per units
for h = 1:Nber
R_min_u{h} = [R_min1(:,h) R_min2(:,h)];
R_max_u{h} = [R_max1(:,h) R_max2(:,h)]; %answers of the experts by moduls/elements
end

%PLUSITER:
%PLUSITER:  (iter=1:... vagyis while, és a konvergencia a leállás...)

maxiter=20;
jj=[];
HHH=1;
iter=1;
for iter=1:maxiter
%while HHH(end)>0
nExp = 2; %number of experts
x = 0+4*eps:0.01:100+4*eps;

g_1=[];
g_4=[];
g_8=[];
g_9=[];
% g_y=[0.1,0.7,0.2;0.1,0.7,0.2;0.1,0.7,0.2;0.1,0.7,0.2];
g_y=[1/3 1/3 1/3;1/3 1/3 1/3;1/3 1/3 1/3;1/3 1/3 1/3];

% g_1_norm{iter}= g_1./sum(g_1,2);
% g_4_norm{iter}= g_4./sum(g_4,2);
% g_8_norm{iter}= g_8./sum(g_8,2);
% g_9_norm{iter}= g_9./sum(g_9,2);
if iter==1
g_y_norm{iter}=g_y./sum(g_y,2); 
end

pdfx_1=[]; %PET
pdfx_4=[];%LDPE
pdfx_6=[];%HDPE
pdfx_7=[];%Haz
pdfx_8=[];%Fe
pdfx_9=[];%Al
pdfx_10=[];%RDF
pdfx_11=[];%Card
pdfx_12=[];%Paper
rp_1=[]; %PET
rp_4=[];%Foil
rp_6=[];%PP
rp_7=[];%Haz
rp_8=[];%Fe
rp_9=[];%Al
rp_10=[];%RDF
rp_11=[];%Card
rp_12=[];%Paper

for h = 2:Nber

for i = 1:nExp
    pd_x1 = makedist('Uniform','Lower',R_min_u{h}(1,i),'Upper',R_max_u{h}(1,i));
    pdfx_1 = [pdfx_1; pdf(pd_x1,x)];
        
    pd_x4 = makedist('Uniform','Lower',R_min_u{h}(4,i),'Upper',R_max_u{h}(4,i));
    pdfx_4 = [pdfx_4; pdf(pd_x4,x)];
        
    pd_x6 = makedist('Uniform','Lower',R_min_u{h}(6,i),'Upper',R_max_u{h}(6,i));
    pdfx_6 = [pdfx_6; pdf(pd_x6,x)];
       
    pd_x7 = makedist('Uniform','Lower',R_min_u{h}(7,i),'Upper',R_max_u{h}(7,i));
    pdfx_7 = [pdfx_7; pdf(pd_x7,x)];
        
    pd_x8 = makedist('Uniform','Lower',R_min_u{h}(8,i),'Upper',R_max_u{h}(8,i));
    pdfx_8 = [pdfx_8; pdf(pd_x8,x)];
        
    pd_x9 = makedist('Uniform','Lower',R_min_u{h}(9,i),'Upper',R_max_u{h}(9,i));
    pdfx_9 = [pdfx_9; pdf(pd_x9,x)];
        
    pd_x10 = makedist('Uniform','Lower',R_min_u{h}(10,i),'Upper',R_max_u{h}(10,i));
    pdfx_10 = [pdfx_10; pdf(pd_x10,x)];
        
    pd_x11 = makedist('Uniform','Lower',R_min_u{h}(11,i),'Upper',R_max_u{h}(11,i));
    pdfx_11 = [pdfx_11; pdf(pd_x11,x)];
        
    pd_x12 = makedist('Uniform','Lower',R_min_u{h}(12,i),'Upper',R_max_u{h}(12,i));
    pdfx_12 = [pdfx_12; pdf(pd_x12,x)];
       
end % for every expert...

pdfx1 = [pdfx_1(end-1,:);pdfx_1(end,:)];
pdfx4 = [pdfx_4(end-1,:);pdfx_4(end,:)];
pdfx6 = [pdfx_6(end-1,:);pdfx_6(end,:)];
pdfx7 = [pdfx_7(end-1,:);pdfx_7(end,:)];
pdfx8 = [pdfx_8(end-1,:);pdfx_8(end,:)];
pdfx9 = [pdfx_9(end-1,:);pdfx_9(end,:)];
pdfx10 = [pdfx_10(end-1,:);pdfx_10(end,:)];
pdfx11 = [pdfx_11(end-1,:);pdfx_11(end,:)];
pdfx12 = [pdfx_12(end-1,:);pdfx_12(end,:)]; %the last two rows separated 

%PLUSITER: generating weighted means of the distributions
if ismember(h,[2 3 4 5 6]) && iter>1  %PET
rp1=sum(pdfx1.*g_1_norm{iter-1}(h,:)'); %weighted mean of the original distributions
else
rp1=mean(pdfx1); %aggregated distribution (mean of the original distribuitons)
end
rp_1=[rp_1;rp1]; %for c1 component, rows corresponds to the units



if ismember(h,[2 3 4 5 7]) && iter>1 %LDPE
rp4=sum(pdfx4.*g_4_norm{iter-1}(h,:)');
else
rp4=mean(pdfx4);%for c4 component, rows corresponds to the units
end
rp_4=[rp_4;rp4];

rp6=mean(pdfx6); %HDPE
rp_6=[rp_6;rp6];
rp7=mean(pdfx7); %Haz
rp_7=[rp_7;rp7];

if h==2 && iter>1 %Fe
rp8=sum(pdfx8.*g_8_norm{iter-1}(h,:)');
else
rp8=mean(pdfx8);
end
rp_8=[rp_8;rp8];

if ismember(h,[2 3]) && iter>1 %Al
rp9=sum(pdfx9.*g_9_norm{iter-1}(h,:)');
else
rp9=mean(pdfx9);
end
rp_9=[rp_9;rp9];

rp10=mean(pdfx10); %RDF
rp_10=[rp_10;rp10];
rp11=mean(pdfx11); %Cardboard
rp_11=[rp_11;rp11];
rp12=mean(pdfx12); %Paper
rp_12=[rp_12;rp12];


if abr==1
figure(h)
subplot(3,3,1)
hold on
plot(x,pdfx1,'LineWidth',1.5)
stairs(x(1:26:end),rp1(1:26:end),'r--','LineWidth',2.5)
xlabel('')
ylabel('Probability density, PET')
legend('Exp.1','Exp.2','Aggregated distribution')

subplot(3,3,2)
hold on
plot(x,pdfx4,'LineWidth',1.5)
stairs(x(1:26:end),rp4(1:26:end),'r--','LineWidth',2.5)
xlabel('')
ylabel('Probability density, LDPE')
legend('Exp.1','Exp.2','Aggregated distribution')

subplot(3,3,3)
hold on
plot(x,pdfx6,'LineWidth',1.5)
stairs(x(1:26:end),rp6(1:26:end),'r--','LineWidth',2.5)
xlabel('')
ylabel('Probability density, HDPE')
legend('Exp.1','Exp.2','Aggregated distribution')

subplot(3,3,4)
hold on
plot(x,pdfx7,'LineWidth',1.5)
stairs(x(1:26:end),rp7(1:26:end),'r--','LineWidth',2.5)
xlabel('')
ylabel('Probability density, Haz')
legend('Exp.1','Exp.2','Aggregated distribution')

subplot(3,3,5)
hold on
plot(x,pdfx8,'LineWidth',1.5)
stairs(x(1:26:end),rp8(1:26:end),'r--','LineWidth',2.5)
xlabel('')
ylabel('Probability density, Iron')
legend('Exp.1','Exp.2','Aggregated distribution')

subplot(3,3,6)
hold on
plot(x,pdfx9,'LineWidth',1.5)
stairs(x(1:26:end),rp9(1:26:end),'r--','LineWidth',2.5)
xlabel('')
ylabel('Probability density, Aluminium')
legend('Exp.1','Exp.2','Aggregated distribution')

subplot(3,3,7)
hold on
plot(x,pdfx10,'LineWidth',1.5)
stairs(x(1:26:end),rp10(1:26:end),'r--','LineWidth',2.5)
xlabel('')
ylabel('Probability density, RDF')
legend('Exp.1','Exp.2','Aggregated distribution')

subplot(3,3,8)
hold on
plot(x,pdfx11,'LineWidth',1.5)
stairs(x(1:26:end),rp11(1:26:end),'r--','LineWidth',2.5)
xlabel('')
ylabel('Probability density, Cardboard')
legend('Exp.1','Exp.2','Aggregated distribution')

subplot(3,3,9)
hold on
plot(x,pdfx12,'LineWidth',1.5)
stairs(x(1:26:end),rp12(1:26:end),'r--','LineWidth',2.5)
xlabel('')
ylabel('Probability density, Paper')
legend('Exp.1','Exp.2','Aggregated distribution')
end

end  %for every unit


%% Generation of random numbers  

px = x;


for i = 1:size(rp_1,1)  %size(rp_1,1)=Nber
rand1(:,i) = randpdf(rp_1(i,:),px,[M,1]); % M: number of MC simulations
rand4(:,i) = randpdf(rp_4(i,:),px,[M,1]);
rand6(:,i) = randpdf(rp_6(i,:),px,[M,1]);
rand7(:,i) = randpdf(rp_7(i,:),px,[M,1]);
rand8(:,i) = randpdf(rp_8(i,:),px,[M,1]);
rand9(:,i) = randpdf(rp_9(i,:),px,[M,1]);
rand10(:,i) = randpdf(rp_10(i,:),px,[M,1]);
rand11(:,i) = randpdf(rp_11(i,:),px,[M,1]);
rand12(:,i) = randpdf(rp_12(i,:),px,[M,1]); % 12 component (or 9 in the present case)
end %the efficiencies for every components and every units

rand_1 = [100*ones(M,1) rand1];
rand_4 = [100*ones(M,1) rand4];
rand_6 = [100*ones(M,1) rand6];
rand_7 = [100*ones(M,1) rand7];
rand_8 = [100*ones(M,1) rand8];
rand_9 = [100*ones(M,1) rand9];
rand_10 = [100*ones(M,1) rand10];
rand_11 = [100*ones(M,1) rand11];
rand_12 = [100*ones(M,1) rand12];

R = cell(1,9);

R{1,1}=rand_1;
R{1,2}=zeros(M,7);
R{1,3}=zeros(M,7);
R{1,4}=rand_4;
R{1,5}=zeros(M,7);
R{1,6}=rand_6;
R{1,7}=rand_7;
R{1,8}=rand_8;
R{1,9}=rand_9;
R{1,10}=rand_10;
R{1,11}=rand_11;
R{1,12}=rand_12;




%% MC simulation

for n = 1 : M
    % Simulation of a given component
    for comp=1:length(xin) %components
        Aw=A;
        for i=1:max(edges(:,1))  %equal to Nber...
            ind=find(A(i,:)==1);
            Aw(i,ind(1))=R{1,comp}(n,i)/100; %change ones to the separation efficiencies (0-1)
            if length(ind)==2
                Aw(i,ind(2))=1-R{1,comp}(n,i)/100;
            end
        end
        %Simulation (Steady state model) 
        u(1)=f11*xin(comp); % kg/h
        x=linsolve(eye(Ns)-Aw',u); %the x flows
        Y(n,1:length(x)-7,comp)=x(8:end,end); %the output flows
        Y(n,length(x)-7+1:length(types)-7,comp)=0;
        
    end
end

%% Validate the result - Más ábrázolás kell majd
%P=[Y(:,6,1)/xin(1) Y(:,5,4)/xin(4) Y(:,1,8)/xin(8) Y(:,2,9)/xin(9)]/f11*100;
P=[Y(:,5,1)/xin(1) Y(:,7,4)/xin(4) Y(:,1,8)/xin(8) Y(:,2,9)/xin(9)]/f11*100;
Exp_min = [83 80 80; 85 80 83; 82 85 85; 90 85 90];
Exp_max = [90 85 83; 90 85 85; 85 90 88; 92 90 93]; %from the experts
markers={'ro','gx','bd'};

if abr==1
figure(20)
%clf
for j=1:4
    for e=1:3 %experts
        plot(j,(Exp_min(j,e)),markers{e},j,(Exp_max(j,e)),markers{e}, 'Markersize',10)
        hold on 
    end
end
boxplot(P)
xticklabels({'PET','LDPE','Iron','Aluminium'})
ylabel('Yield [%]')

figure(21)
for j=1:4
subplot(4,1,j)
histogram(P(:,j))
end
end

%For comparison
if iter==1
figure(100)
for j=1:4
subplot(4,1,j)
ksdensity(P(:,j))
hold on
end
end

%% Expert-based distribution - Results
nE = 3; %number of experts
x = [75+4*eps:0.01:95];

%clf
pdfE=[];
pdfL=[];
pdfI=[];
pdfA=[];
for i=1:nE
pd_PET = makedist('Uniform','Lower',Exp_min(1,i),'Upper',Exp_max(1,i));
pdfE=[pdfE; pdf(pd_PET,x)];

pd_LDPE = makedist('Uniform','Lower',Exp_min(2,i),'Upper',Exp_max(2,i));
pdfL = [pdfL; pdf(pd_LDPE,x)];

pd_I = makedist('Uniform','Lower',Exp_min(3,i),'Upper',Exp_max(3,i));
pdfI = [pdfI; pdf(pd_I,x)];

pd_Al = makedist('Uniform','Lower',Exp_min(4,i),'Upper',Exp_max(4,i));
pdfA = [pdfA; pdf(pd_Al,x)];

end

if abr==0
figure(31)
plot(x,pdfE,'LineWidth',1.5)
hold on 
rp20=mean(pdfE);
stairs(x(1:26:end),rp20(1:26:end),'r--','LineWidth',2.5)
xlabel('Yield of PET in x_{11} [%]')
ylabel('Probability density')
legend('Exp.1','Exp.2','Exp.3','Aggregated distribution')

figure(32)
plot(x,pdfL,'LineWidth',1.5)
hold on 
rp21=mean(pdfL);
stairs(x(1:26:end),rp21(1:26:end),'r--','LineWidth',2.5)
xlabel('Yield of LDPE in x_{12} [%]')
ylabel('Probability density')
legend('Exp.1','Exp.2','Exp.3','Aggregated distribution')

figure(33)
plot(x,pdfI,'LineWidth',1.5)
hold on 
rp23=mean(pdfI);
stairs(x(1:26:end),rp23(1:26:end),'r--','LineWidth',2.5)
xlabel('Yield of Iron in x_{7} [%]')
ylabel('Probability density')
legend('Exp.1','Exp.2','Exp.3','Aggregated distribution')

figure(34)
plot(x,pdfA,'LineWidth',1.5)
hold on 
rp24=mean(pdfA);
stairs(x(1:32:end),rp24(1:32:end),'r--','LineWidth',2.5)
xlabel('Yield of ALuminium in x_{8} [%]')
ylabel('Probability density')
legend('Exp.1','Exp.2','Exp.3','Aggregated distribution')
end


%% Importance sampling (all)

Yout=cell(1,size(P,2)); %Yout{iteráció,1 (comp lehet majd, ha úgy van...?)}
w=cell(1,size(P,2));
%P= yield'PET','LDPE','Iron','Aluminium'
compout=[1 4 8 9];

for i=1:size(P,2)
Yout{1,i}=P(:,i); %output at the 1st iteration
end

j=1;
H=1;
H1=1;
H2=1;
HH=1;
while HH>0 %K-S test for stop the iteration
    
%for j=1:20 %with fix iteration number

%R{iteráció,comp=Al}(MC,r1(Al))
for i=1:size(R,2) %update rest of the r-s
R{j+1,i}=R{j,i};
end

for k=1:size(P,2) %for all 4 output, resamle the r-s
w1=zeros(M,nE);

for i=1:nE
    if k==1
    pd_PET = makedist('Uniform','Lower',Exp_min(1,i),'Upper',Exp_max(1,i));
    w1(:,i)=pdf(pd_PET,Yout{j,k}); %p(x)

    elseif k==2
    pd_LDPE = makedist('Uniform','Lower',Exp_min(2,i),'Upper',Exp_max(2,i));
    w1(:,i)=pdf(pd_LDPE,Yout{j,k});
    
    elseif k==3
    pd_I = makedist('Uniform','Lower',Exp_min(3,i),'Upper',Exp_max(3,i));
    w1(:,i)=pdf(pd_I,Yout{j,k});

    elseif k==4
    pd_Al = makedist('Uniform','Lower',Exp_min(4,i),'Upper',Exp_max(4,i));
    w1(:,i)=pdf(pd_Al,Yout{j,k}); %evaluate the pdf at the given output points
    end
end

w2=ksdensity(Yout{j,k}',Yout{j,k}');
%w{j,k}=mean(w1,2); %only p(x) részecskék súlyozása az output alapján


%PLUSITER: weighted sum of the outputs
if iter>=1
w{j,k}=sum(w1'.*g_y_norm{iter}(k,:)')./w2;
%w{j,k}=mean(w1,2)'./w2;
else
w{j,k}=mean(w1,2)'./w2; %use p(x)/q(x) importance weights
end

if k==1 %R{iteration, component}(particle,unit)
[R{j+1,1}(:,2)]=resample(R{j,1}(:,2)',w{j,1}, 'multinomial_resampling') %resampling(az r1 Al oszlopa elõször, w1, 'systematic resampling')
[R{j+1,1}(:,3)]=resample(R{j,1}(:,3)',w{j,1}, 'multinomial_resampling') %resampling(az r2 Al oszlopa is, w1, 'systematic resampling')
[R{j+1,1}(:,4)]=resample(R{j,1}(:,4)',w{j,1}, 'multinomial_resampling') %resampling(az r1 Al oszlopa elõször, w1, 'systematic resampling')
[R{j+1,1}(:,5)]=resample(R{j,1}(:,5)',w{j,1}, 'multinomial_resampling') %resampling(az r2 Al oszlopa is, w1, 'systematic resampling')
[R{j+1,1}(:,6)]=resample(R{j,1}(:,6)',w{j,1}, 'multinomial_resampling') %resampling(az r1 Al oszlopa elõször, w1, 'systematic resampling')

elseif k==2
[R{j+1,4}(:,2)]=resample(R{j,4}(:,2)',w{j,2}, 'multinomial_resampling') %resampling(az r1 Al oszlopa elõször, w1, 'systematic resampling')
[R{j+1,4}(:,3)]=resample(R{j,4}(:,3)',w{j,2}, 'multinomial_resampling') %resampling(az r2 Al oszlopa is, w1, 'systematic resampling')
[R{j+1,4}(:,4)]=resample(R{j,4}(:,4)',w{j,2}, 'multinomial_resampling') %resampling(az r1 Al oszlopa elõször, w1, 'systematic resampling')
[R{j+1,4}(:,5)]=resample(R{j,4}(:,5)',w{j,2}, 'multinomial_resampling') %resampling(az r2 Al oszlopa is, w1, 'systematic resampling')
[R{j+1,4}(:,7)]=resample(R{j,4}(:,7)',w{j,2}, 'multinomial_resampling') %resampling(az r1 Al oszlopa elõször, w1, 'systematic resampling')

elseif k==3
[R{j+1,8}(:,2)]=resample(R{j,8}(:,2)',w{j,3}, 'multinomial_resampling') %resampling(az r1 Al oszlopa elõször, w1, 'systematic resampling')

elseif k==4
[R{j+1,9}(:,2)]=resample(R{j,9}(:,2)',w{j,4}, 'multinomial_resampling') %resampling(az r1 Al oszlopa elõször, w1, 'systematic resampling')
[R{j+1,9}(:,3)]=resample(R{j,9}(:,3)',w{j,4}, 'multinomial_resampling') %resampling(az r2 Al oszlopa is, w1, 'systematic resampling')
end
end

for n = 1 : M   %x, majd Y újraszámol (itt már megvan az 5000 részecske, csak r2 és r3 frissítése az Al-nél)
    % Simulation of a given component
    for comp=1:length(xin) %components
        Aw=A;
        for i=1:max(edges(:,1))  %equal to Nber...
            ind=find(A(i,:)==1);
            Aw(i,ind(1))=R{j+1,comp}(n,i)/100; %change ones to the separation efficiencies (0-1)
            if length(ind)==2
                Aw(i,ind(2))=1-R{j+1,comp}(n,i)/100;
            end
        end
        %Simulation (Steady state model) 
        u(1)=f11*xin(comp); % kg/h
        x=linsolve(eye(Ns)-Aw',u);
        Y(n,1:length(x)-7,comp)=x(8:end,end); %the output flows
        Y(n,length(x)-7+1:length(types)-7,comp)=0;
    end
end

P=[Y(:,5,1)/xin(1) Y(:,7,4)/xin(4) Y(:,1,8)/xin(8) Y(:,2,9)/xin(9)]/f11*100;

for l=1:size(P,2)
Yout{j+1,l}=P(:,l); %output at the l-st iteration
end

alfa=0.2;

H_E=kstest2(Yout{j+1,1},Yout{j,1},'alpha',alfa); %with K-S test
H_L=kstest2(Yout{j+1,2},Yout{j,2},'alpha',alfa);
H_I=kstest2(Yout{j+1,3},Yout{j,3},'alpha',alfa);
H_Al=kstest2(Yout{j+1,4},Yout{j,4},'alpha',alfa);
H(1)=H_E+H_L+H_I+H_Al;

H(2)=kstest2(R{j+1,1}(:,2),R{j,1}(:,2),'alpha',alfa);
H(3)=kstest2(R{j+1,1}(:,3),R{j,1}(:,3),'alpha',alfa);
H(4)=kstest2(R{j+1,1}(:,4),R{j,1}(:,4),'alpha',alfa);
H(5)=kstest2(R{j+1,1}(:,5),R{j,1}(:,5),'alpha',alfa);
H(6)=kstest2(R{j+1,1}(:,6),R{j,1}(:,6),'alpha',alfa);

H(7)=kstest2(R{j+1,4}(:,2),R{j,4}(:,2),'alpha',alfa);
H(8)=kstest2(R{j+1,4}(:,3),R{j,4}(:,3),'alpha',alfa);
H(9)=kstest2(R{j+1,4}(:,4),R{j,4}(:,4),'alpha',alfa);
H(10)=kstest2(R{j+1,4}(:,5),R{j,4}(:,5),'alpha',alfa);
H(11)=kstest2(R{j+1,4}(:,7),R{j,4}(:,7),'alpha',alfa);

H(12)=kstest2(R{j+1,8}(:,2),R{j,8}(:,2),'alpha',alfa);

H(13)=kstest2(R{j+1,9}(:,2),R{j,9}(:,2),'alpha',alfa);
H(14)=kstest2(R{j+1,9}(:,3),R{j,9}(:,3),'alpha',alfa);

HH=sum(H);

j=j+1; %step one at the iteration number

% figure %check of the resampling
% ecdf(R{j,9}(:,3)','Frequency',round(w{j,1}(:,1)*1000))
% hold on
% ecdf(R{j,9}(:,3)')
% hold on
% ecdf(R{j+1,9}(:,3)')
% xlim([0 100])

end %and again... until it converges...

%PLUSITER:
%RR and YoutYout are cell arrays! RR{iter,component}(M,unit)
if iter==1
    YoutYout=cell(1,size(Yout,2));
    RR=cell(1,size(R,2));
end

jj(iter)=j %save the required iteration number
RR(iter,:)=R(end,:); %save the converged parameters/outputs on the inside circle
YoutYout(iter,:)=Yout(end,:);

%%
if abr==1
figure
for i=1:size(R,1)
ecdf(R{i,9}(:,2)')
hold on
end
xlim([90 100])
legend
title('r_2^{Al} along the iteration steps')
%% 
fig=figure
for i=1:size(R,1)
ecdf(R{i,9}(:,3)')
hold on
end
xlim([0 10])
%title('r_3^{Al} along the iteration steps')
xlabel('r_3^{Aluminium} [%]')
ylabel('Empirical cdf')
pos=get(gca,'Position');
pos(1)=pos(1)-0.04;
set(gca,'Position',pos);

% Create discrete colorbar

a=1:size(Yout,1);
map = flipud(copper(length(a)));
colororder(flipud(copper(size(Yout,1))));

h = axes(fig,'visible','off'); 
h.Title.Visible = 'on';
h.XLabel.Visible = 'on';
h.YLabel.Visible = 'on';
colormap(map)
hh = colorbar;
tk = linspace(0,1,2*length(a)+1);
%set(hh, 'YTick',tk(2:2:end),'YTickLabel', a','Position',[0.93 0.168 0.022 0.7]);
set(hh, 'YTick',tk(2:2:end),'YTickLabel', a','Position',[0.93 0.168 0.022 0.7]);
hh.Label.String = 'Number of iterations';
pos=get(hh,'Position');
pos(1)=pos(1)-0.03;
set(hh,'Position',pos);
end

%% Colormap???
if abr==1
a=1:size(Yout,1);
% Specify axes ColorOrder
map = flipud(parula(length(a)));
axes('ColorOrder',map) %,'NextPlot','replacechildren')

figure


for i=1:size(Yout,1)
ecdf(Yout{i,1}')
hold on
end

xlim([70 100])
xlabel('Yield, PET [%]')
ylabel('Empirical cdf')

% % Create discrete colorbar
% colormap(map)
% h = colorbar;
% tk = linspace(0,1,2*length(a)+1);
% set(h, 'YTick',tk(2:2:end),'YTickLabel', a');
% h.Label.String = 'Number of iterations';
end
%% Convergence (fig13b)
ii=1
if abr==1
fig=figure


subplot(2,2,1)
for i=ii:size(Yout,1) 
ecdf(Yout{i,1}')
hold on
end
xlim([70 100])
xlabel('Yield, PET [%]')
ylabel('Empirical cdf')
% set(gca,'Position',[0.100    0.5838    0.3311    0.3412])
pos=get(gca,'Position');
pos(1)=pos(1)-0.04;
set(gca,'Position',pos);


subplot(2,2,2)
for i=ii:size(Yout,1)
ecdf(Yout{i,2}')
hold on
end
xlim([70 100])
xlabel('Yield, LDPE [%]')
ylabel('Empirical cdf')
pos=get(gca,'Position');
pos(1)=pos(1)-0.04;
set(gca,'Position',pos);

subplot(2,2,3)
for i=ii:size(Yout,1)
ecdf(Yout{i,3}')
hold on
end
xlim([80 100])
xlabel('Yield, Iron [%]')
ylabel('Empirical cdf')
pos=get(gca,'Position');
pos(1)=pos(1)-0.04;
set(gca,'Position',pos);

subplot(2,2,4)
for i=ii:size(Yout,1)
ecdf(Yout{i,4}')
hold on
end
xlim([80 100])
xlabel('Yield, Aluminium [%]')
ylabel('Empirical cdf')
pos=get(gca,'Position');
pos(1)=pos(1)-0.04;
set(gca,'Position',pos);


% Create discrete colorbar

a=1:size(Yout,1);
map = flipud(copper(length(a)));
colororder(flipud(copper(size(Yout,1))));

h = axes(fig,'visible','off'); 
h.Title.Visible = 'on';
h.XLabel.Visible = 'on';
h.YLabel.Visible = 'on';
colormap(map)
hh = colorbar;
tk = linspace(0,1,2*length(a)+1);
%set(hh, 'YTick',tk(2:2:end),'YTickLabel', a','Position',[0.93 0.168 0.022 0.7]);
set(hh, 'YTick',tk(2:2:end),'YTickLabel', a','Position',[0.93 0.168 0.022 0.7]);
hh.Label.String = 'Number of iterations';
pos=get(hh,'Position');
pos(1)=pos(1)-0.03;
set(hh,'Position',pos);

%sgtitle('Output along the iteration steps')
end
%% Convergence pdf (fig16b)


ii=1;

fig=figure
if abr==1
subplot(2,2,1)
for i=ii:size(Yout,1)
ksdensity(Yout{i,1}')
hold on
end
xlim([70 100])
xlabel('Yield, PET [%]')
ylabel('Empirical pdf')
pos=get(gca,'Position');
pos(1)=pos(1)-0.04;
set(gca,'Position',pos);

subplot(2,2,2)
for i=ii:size(Yout,1)
ksdensity(Yout{i,2}')
hold on
end
xlim([70 100])
xlabel('Yield, LDPE [%]')
ylabel('Empirical pdf')
pos=get(gca,'Position');
pos(1)=pos(1)-0.04;
set(gca,'Position',pos);

subplot(2,2,3)
for i=ii:size(Yout,1)
ksdensity(Yout{i,3}')
hold on
end
xlim([80 100])
xlabel('Yield, Iron [%]')
ylabel('Empirical pdf')
pos=get(gca,'Position');
pos(1)=pos(1)-0.04;
set(gca,'Position',pos);

subplot(2,2,4)
for i=ii:size(Yout,1)
ksdensity(Yout{i,4}')
hold on
end
xlim([80 100])
xlabel('Yield, Aluminium [%]')
ylabel('Empirical pdf')
pos=get(gca,'Position');
pos(1)=pos(1)-0.04;
set(gca,'Position',pos);


% Create discrete colorbar
a=1:size(Yout,1);
map = flipud(copper(length(a)));
colororder(flipud(copper(size(Yout,1))));

hhh = axes(fig,'visible','off'); 
hhh.Title.Visible = 'on';
hhh.XLabel.Visible = 'on';
hhh.YLabel.Visible = 'on';
colormap(map)
hh = colorbar;
tk = linspace(0,1,2*length(a)+1);
set(hh, 'YTick',tk(2:2:end),'YTickLabel', a','Position',[0.93 0.168 0.022 0.7]);
hh.Label.String = 'Number of iterations';
pos=get(hh,'Position');
pos(1)=pos(1)-0.03;
set(hh,'Position',pos);

end
%%
if abr==1
    figure
    %clf
    for j=1:size(P,2)
        for e=1:3 %experts
            plot(j,(Exp_min(j,e)),markers{e},j,(Exp_max(j,e)),markers{e}, 'Markersize',10)
            hold on 
        end
    end
    boxplot(P)
    xticklabels({'PET','LDPE','Iron','Aluminium'})
    %xticklabels({'Aluminium'})
    ylabel('Yield [%]')
    ylim([78 100])
    
    figure
    for j=1:size(P,2)
    subplot(size(P,2),1,j)
    histogram(P(:,j))
    end  
end

%For comparison
if iter==1
figure(100)
for j=1:4
subplot(4,1,j)
ksdensity(Yout{end,j})
end
end
%% Estimated vs. expert-based pdf (r2, r3, Yout(Al))
if abr==1
%PET (c1)
figure
xii = 0+4*eps:0.01:100+4*eps;
subplot(6,1,1)
[f1 xi1]=ksdensity(R{end,1}(:,2)',xii);
plot(xi1,f1,'-r','LineWidth',1) %r2 estimated
hold on
plot(xii,rp_1(1,:),'b','LineWidth',1) %r2 expert
xlim([80 100])
%xlabel('Separation efficiency of the magnetic separator for PET (r_{e_2}(PET)) [%]')
xlabel('r_2^{PET} [%]')
ylabel('Probability density')
legend('Estimated distribution','Expert-based aggregated distribution')

subplot(6,1,2)
[f1 xi1]=ksdensity(R{end,1}(:,3)',xii);
plot(xi1,f1,'-r','LineWidth',1) %r3 estimated
hold on
plot(xii,rp_1(2,:),'b','LineWidth',1) %r3 expert
xlim([80 100])
%xlabel('Separation efficiency of e_3 unit for PET (r_{e_3}(PET)) [%]')
xlabel('r_3^{PET} [%]')
ylabel('Probability density')

subplot(6,1,3)
[f1 xi1]=ksdensity(R{end,1}(:,4)',xii);
plot(xi1,f1,'-r','LineWidth',1) %r4 estimated
hold on
plot(xii,rp_1(3,:),'b','LineWidth',1) %r4 expert
xlim([80 100])
%xlabel('Separation efficiency of e_4 unit for PET (r_{e_4}(PET)) [%]')
xlabel('r_4^{PET} [%]')
ylabel('Probability density')

subplot(6,1,4)
[f1 xi1]=ksdensity(R{end,1}(:,5)',xii);
plot(xi1,f1,'-r','LineWidth',1) %r5 estimated
hold on
plot(xii,rp_1(4,:),'b','LineWidth',1) %r5 expert (rp_1(4,..) because pre-shredder (r(e1)=100%) is not in rp_1
xlim([80 100])
%xlabel('Separation efficiency of e_5 unit for PET (r_{e_5}(PET)) [%]')
xlabel('r_5^{PET} [%]')
ylabel('Probability density')

subplot(6,1,5)
[f1 xi1]=ksdensity(R{end,1}(:,6)',xii);
plot(xi1,f1,'-r','LineWidth',1) %r6 estimated
hold on
plot(xii,rp_1(5,:),'b','LineWidth',1) %r6 expert
xlim([0 20])
%xlabel('Separation efficiency of e_6 unit for PET (r_{e_6}(PET)) [%]')
xlabel('r_6^{PET} [%]')
ylabel('Probability density')

subplot(6,1,6)
xiii = [75+4*eps:0.01:95];
[f1 xi1]=ksdensity(Yout{end,1}',xii);
plot(xi1,f1,'-r','LineWidth',1) %Yout estimated
hold on
rp24=mean(pdfE);
stairs(xiii(1:26:end),rp24(:,1:26:end),'b-','LineWidth',1) %Yout expert
xlabel('Yield of PET in x_{12} [%]')
ylabel('Probability density')
xlim([75 95])

end
%% LDPE (c4)
if abr==1
figure
xii = 0+4*eps:0.01:100+4*eps;
subplot(6,1,1)
[f1 xi1]=ksdensity(R{end,4}(:,2)');
plot(xi1,f1,'-r','LineWidth',1) %r2 estimated
hold on
plot(xii,rp_4(1,:),'b','LineWidth',1) %r2 expert
xlim([80 100])
%xlabel('Separation efficiency of e_2 unit for LDPE (r_{e_2}(LDPE)) [%]')
xlabel('r_2^{LDPE} [%]')
ylabel('Probability density')
legend('Estimated distribution','Expert-based aggregated distribution')

subplot(6,1,2)
[f1 xi1]=ksdensity(R{end,4}(:,3)');
plot(xi1,f1,'-r','LineWidth',1) %r3 estimated
hold on
plot(xii,rp_4(2,:),'b','LineWidth',1) %r3 expert
xlim([80 100])
%xlabel('Separation efficiency of e_3 unit for LDPE (r_{e_3}(LDPE)) [%]')
xlabel('r_3^{LDPE} [%]')
ylabel('Probability density')

subplot(6,1,3)
[f1 xi1]=ksdensity(R{end,4}(:,4)');
plot(xi1,f1,'-r','LineWidth',1) %r4 estimated
hold on
plot(xii,rp_4(3,:),'b','LineWidth',1) %r4 expert
xlim([80 100])
%xlabel('Separation efficiency of e_4 unit for LDPE (r_{e_4}(LDPE)) [%]')
xlabel('r_4^{LDPE} [%]')
ylabel('Probability density')

subplot(6,1,4)
[f1 xi1]=ksdensity(R{end,4}(:,5)');
plot(xi1,f1,'-r','LineWidth',1) %r5 estimated
hold on
plot(xii,rp_4(4,:),'b','LineWidth',1) %r5 expert
xlim([0 20])
%xlabel('Separation efficiency of e_5 unit for LDPE (r_{e_5}(LDPE)) [%]')
xlabel('r_5^{LDPE} [%]')
ylabel('Probability density')

subplot(6,1,5)
[f1 xi1]=ksdensity(R{end,4}(:,7)');
plot(xi1,f1,'-r','LineWidth',1) %r7 estimated
hold on
plot(xii,rp_4(6,:),'b','LineWidth',1) %r7 expert
xlim([0 20])
%xlabel('Separation efficiency of e_7 unit for LDPE (r_{e_7}(LDPE)) [%]')
xlabel('r_7^{LDPE} [%]')
ylabel('Probability density')

subplot(6,1,6)
xiii = [75+4*eps:0.01:95];
[f1 xi1]=ksdensity(Yout{end,2}');
plot(xi1,f1,'-r','LineWidth',1) %Yout estimated
hold on
rp24=mean(pdfL);
stairs(xiii(1:26:end),rp24(1:26:end),'b-','LineWidth',1) %Yout expert
xlabel('Yield of LDPE in m_{14} [%]')
ylabel('Probability density')


%Vas (c8)
figure
subplot(2,1,1)
xii = 0+4*eps:0.01:100+4*eps;
[f1 xi1]=ksdensity(R{end,8}(:,2)');
plot(xi1,f1,'-r','LineWidth',1) %r2 estimated
hold on
plot(xii,rp_8(1,:),'b','LineWidth',1) %r2 expert
xlim([0 20])
%xlabel('Separation efficiency of e_2 unit for Iron (r_{e_2}(Fe)) [%]')
xlabel('r_2^{Iron} [%]')
ylabel('Probability density')
legend('Estimated distribution','Expert-based aggregated distribution')



subplot(2,1,2)
xiii = [75+4*eps:0.01:95];
[f1 xi1]=ksdensity(Yout{end,3}');
plot(xi1,f1,'-r','LineWidth',1) %Yout estimated
hold on
rp24=mean(pdfI);
stairs(xiii(1:26:end),rp24(1:26:end),'b-','LineWidth',1) %Yout expert
xlabel('Yield of Iron in x_{8} [%]')
ylabel('Probability density')

%Aluminium (c9)
figure
xii = 0+4*eps:0.01:100+4*eps;
subplot(3,1,1)
[f1 xi1]=ksdensity(R{end,9}(:,2)');
plot(xi1,f1,'-r','LineWidth',1) %r2 estimated
hold on
stairs(xii(1:13:end),rp_9(1,1:13:end),'b','LineWidth',1) %r2 expert
xlim([90 100])
%xlabel('Separation efficiency of e_2 unit for Aluminium (r_{e_2}(Al)) [%]')
xlabel('r_2^{Aluminium} [%]')
ylabel('Probability density')
legend('Estimated distribution','Expert-based aggregated distribution')

subplot(3,1,2)
[f1 xi1]=ksdensity(R{end,9}(:,3)');
plot(xi1,f1,'-r','LineWidth',1) %r3 estimated
hold on
plot(xii,rp_9(2,:),'b','LineWidth',1) %r3 expert
xlim([0 15])
%xlabel('Separation efficiency of e_3 unit for Aluminium (r_{e_3}(Al)) [%]')
xlabel('r_3^{Aluminium} [%]')
ylabel('Probability density')
end

xiii = [75+4*eps:0.01:95];

if abr==1
subplot(3,1,3)
[f1 xi1]=ksdensity(Yout{end,4}');
plot(xi1,f1,'-r','LineWidth',1) %Yout estimated
hold on
rp24=mean(pdfA);
stairs(xiii(1:26:end),rp24(1:26:end),'b-','LineWidth',1) %Yout expert
xlabel('Yield of Aluminium in x_{9} [%]')
ylabel('Probability density')
end
%% Evaluation of the experts
% Gathering expert based distributions (parameters)
%% Evaluation of the experts
% Gathering expert based distributions (parameters)
xii = 0+4*eps:0.01:100+4*eps;

for h = 2:Nber
for j = 1:nExp
    %PET
    if ismember(h,[2 3 4 5 6])
    pd_x1 = makedist('Uniform','Lower',R_min_u{h}(1,j),'Upper',R_max_u{h}(1,j));
    p_expert{h,j,1}=pdf(pd_x1,xii);
    [f1 xi1]=ksdensity(R{end,1}(:,h)',xii); %f1: estimated probability
    g_1(h,j)=sum(min(p_expert{h,j,1},f1))/sum(max(p_expert{h,j,1},f1));
    end

    %LDPE
    if ismember(h,[2 3 4 5 7])
    pd_x4 = makedist('Uniform','Lower',R_min_u{h}(4,j),'Upper',R_max_u{h}(4,j));
    p_expert{h,j,4}=pdf(pd_x4,xii);
    [f1 xi1]=ksdensity(R{end,4}(:,h)',xii);
    g_4(h,j)=sum(min(p_expert{h,j,4},f1))/sum(max(p_expert{h,j,4},f1)); 
    end

    %Iron  
    if ismember(h,[2])
    pd_x8 = makedist('Uniform','Lower',R_min_u{h}(8,j),'Upper',R_max_u{h}(8,j));
    p_expert{h,j,8}=pdf(pd_x8,xii);
    [f1 xi1]=ksdensity(R{end,8}(:,h)',xii);
    g_8(h,j)=sum(min(p_expert{h,j,8},f1))/sum(max(p_expert{h,j,8},f1));
    end
        
    %Aluminium
    if ismember(h,[2 3])
    pd_x9 = makedist('Uniform','Lower',R_min_u{h}(9,j),'Upper',R_max_u{h}(9,j));
    p_expert{h,j,9}=pdf(pd_x9,xii);
    [f1 xi1]=ksdensity(R{end,9}(:,h)',xii); 
    g_9(h,j)=sum(min(p_expert{h,j,9},f1))/sum(max(p_expert{h,j,9},f1));
    end
         
end % for every expert...
end

%%
for j=1:nE
pd_PET = makedist('Uniform','Lower',Exp_min(1,j),'Upper',Exp_max(1,j));
y_expert{j,1}=pdf(pd_PET,xii);
[f1 xi1]=ksdensity(Yout{end,1}',xii);
g_y(1,j)=sum(min(y_expert{j,1},f1))/sum(max(y_expert{j,1},f1));


pd_LDPE = makedist('Uniform','Lower',Exp_min(2,j),'Upper',Exp_max(2,j));
y_expert{j,2}=pdf(pd_LDPE,xii);
[f1 xi1]=ksdensity(Yout{end,2}',xii);
g_y(2,j)=sum(min(y_expert{j,2},f1))/sum(max(y_expert{j,2},f1));

pd_I = makedist('Uniform','Lower',Exp_min(3,j),'Upper',Exp_max(3,j));
y_expert{j,3}=pdf(pd_I,xii);
[f1 xi1]=ksdensity(Yout{end,3}',xii);
g_y(3,j)=sum(min(y_expert{j,3},f1))/sum(max(y_expert{j,3},f1));

pd_Al = makedist('Uniform','Lower',Exp_min(4,j),'Upper',Exp_max(4,j));
y_expert{j,4}=pdf(pd_Al,xii);
[f1 xi1]=ksdensity(Yout{end,4}',xii);
g_y(4,j)=sum(min(y_expert{j,4},f1))/sum(max(y_expert{j,4},f1))
end

%PLUSITER:
%Normalizing the goodness values (giving weights to the expert opinions)
g_1_norm{iter}= g_1./sum(g_1,2);
g_4_norm{iter}= g_4./sum(g_4,2);
g_8_norm{iter}= g_8./sum(g_8,2);
g_9_norm{iter}= g_9./sum(g_9,2);
g_y_norm{iter+1}=g_y./sum(g_y,2); 

if iter>1
alfa=0.01;

H_E=kstest2(YoutYout{iter-1,1},YoutYout{iter,1},'alpha',alfa); %with K-S test
H_L=kstest2(YoutYout{iter-1,2},YoutYout{iter,2},'alpha',alfa);
H_I=kstest2(YoutYout{iter-1,3},YoutYout{iter,3},'alpha',alfa);
H_Al=kstest2(YoutYout{iter-1,4},YoutYout{iter,4},'alpha',alfa);
H(1)=H_E+H_L+H_I+H_Al;

H(2)=kstest2(RR{iter-1,1}(:,2),RR{iter,1}(:,2),'alpha',alfa);
H(3)=kstest2(RR{iter-1,1}(:,3),RR{iter,1}(:,3),'alpha',alfa);
H(4)=kstest2(RR{iter-1,1}(:,4),RR{iter,1}(:,4),'alpha',alfa);
H(5)=kstest2(RR{iter-1,1}(:,5),RR{iter,1}(:,5),'alpha',alfa);
H(6)=kstest2(RR{iter-1,1}(:,6),RR{iter,1}(:,6),'alpha',alfa);

H(7)=kstest2(RR{iter-1,4}(:,2),RR{iter,4}(:,2),'alpha',alfa);
H(8)=kstest2(RR{iter-1,4}(:,3),RR{iter,4}(:,3),'alpha',alfa);
H(9)=kstest2(RR{iter-1,4}(:,4),RR{iter,4}(:,4),'alpha',alfa);
H(10)=kstest2(RR{iter-1,4}(:,5),RR{iter,4}(:,5),'alpha',alfa);
H(11)=kstest2(RR{iter-1,4}(:,7),RR{iter,4}(:,7),'alpha',alfa);

H(12)=kstest2(RR{iter-1,8}(:,2),RR{iter,8}(:,2),'alpha',alfa);

H(13)=kstest2(RR{iter-1,9}(:,2),RR{iter,9}(:,2),'alpha',alfa);
H(14)=kstest2(RR{iter-1,9}(:,3),RR{iter,9}(:,3),'alpha',alfa);

HHH(iter)=sum(H)
end
iter=iter+1;
end
ttt=toc;

%For comparison
figure(100)
for j=1:4
subplot(4,1,j)
ksdensity(Yout{end,j})
end

x = [75+4*eps:0.01:95];
figure(100)
subplot(4,1,1)
stairs(x(1:26:end),rp20(1:26:end),'k--','LineWidth',0.5)
subplot(4,1,2)
stairs(x(1:26:end),rp21(1:26:end),'k--','LineWidth',0.5)
subplot(4,1,3)
stairs(x(1:26:end),rp23(1:26:end),'k--','LineWidth',0.5)
subplot(4,1,4)
stairs(x(1:32:end),rp24(1:32:end),'k--','LineWidth',0.5)
hold on

figure(100)
xlabels={'Yield of PET in x_{11} [%]','Yield of LDPE in x_{12} [%]','Yield of Iron in x_{7} [%]','Yield of Aluminium in x_{8} [%]'}
for j=1:4
subplot(4,1,j)
xlabel(xlabels(j))
ylabel('Probability density [-]')
xlim([60 100])
legend('Aggregated expert knowledge','MC simulation','MC-IS without outer circle','MC-IS with outer circle','Location','northwest')
end


%% QQplot r2LDPE, Expert 2.
pd_x4 = makedist('Uniform','Lower',R_min_u{2}(4,2),'Upper',R_max_u{2}(4,2));
   
figure
qqplot(R{end,4}(:,2)',pd_x4)
xlabel('Quantiles of Uniform Distribution from Expert 2. (r_2^{LDPE}) [-]')
ylabel('Quantiles of the Empirical Distribution (r_2^{LDPE}) [-]')
title('')

% figure
% plot(g_y,g_y,'.',g_1([2 3 4 5 6],:),g_1([2 3 4 5 6],:),'.',g_4([2 3 4 5 7],:),g_4([2 3 4 5 7],:),'.',g_8(2,:),g_8(2,:),'.',g_9(2:3,:),g_9(2:3,:),'.')
% qqplot(Yout{end,4}',pd_Al)

%% Convergence plots at the outer iteration circle 

%r_3^{Aluminium}
fig=figure
for i=1:size(RR,1)
ecdf(RR{i,9}(:,3)')
hold on
end
xlim([0 10])
%title('r_3^{Al} along the iteration steps')
xlabel('r_3^{Aluminium} [%]')
ylabel('Empirical cdf')
pos=get(gca,'Position');
pos(1)=pos(1)-0.04;
set(gca,'Position',pos);

% Create discrete colorbar

a=1:size(YoutYout,1);
map = flipud(parula(length(a)));
colororder(flipud(parula(size(YoutYout,1))));

h = axes(fig,'visible','off'); 
h.Title.Visible = 'on';
h.XLabel.Visible = 'on';
h.YLabel.Visible = 'on';
colormap(map)
hh = colorbar;
tk = linspace(0,1,2*length(a)+1);
%set(hh, 'YTick',tk(2:2:end),'YTickLabel', a','Position',[0.93 0.168 0.022 0.7]);
set(hh, 'YTick',tk(2:2:end),'YTickLabel', a','Position',[0.93 0.168 0.022 0.7]);
hh.Label.String = 'Number of iterations';
pos=get(hh,'Position');
pos(1)=pos(1)-0.03;
set(hh,'Position',pos);

%% Outputs
ii=1

if abr==0
    
fig=figure


subplot(2,2,1)
for i=ii:size(YoutYout,1) 
ecdf(YoutYout{i,1}')
hold on
end
xlim([70 100])
xlabel('Yield, PET [%]')
ylabel('Empirical cdf [-]')
pos=get(gca,'Position');
pos(1)=pos(1)-0.04;
set(gca,'Position',pos);


subplot(2,2,2)
for i=ii:size(YoutYout,1)
ecdf(YoutYout{i,2}')
hold on
end
xlim([70 100])
xlabel('Yield, LDPE [%]')
ylabel('Empirical cdf [-]')
pos=get(gca,'Position');
pos(1)=pos(1)-0.04;
set(gca,'Position',pos);

subplot(2,2,3)
for i=ii:size(YoutYout,1)
ecdf(YoutYout{i,3}')
hold on
end
xlim([80 100])
xlabel('Yield, Iron [%]')
ylabel('Empirical cdf [-]')
pos=get(gca,'Position');
pos(1)=pos(1)-0.04;
set(gca,'Position',pos);

subplot(2,2,4)
for i=ii:size(YoutYout,1)
ecdf(YoutYout{i,4}')
hold on
end
xlim([80 100])
xlabel('Yield, Aluminium [%]')
ylabel('Empirical cdf [-]')
pos=get(gca,'Position');
pos(1)=pos(1)-0.04;
set(gca,'Position',pos);


% Create discrete colorbar

a=1:size(YoutYout,1);
map = flipud(copper(length(a)));
colororder(flipud(copper(size(YoutYout,1))));

h = axes(fig,'visible','off'); 
h.Title.Visible = 'on';
h.XLabel.Visible = 'on';
h.YLabel.Visible = 'on';
colormap(map)
hh = colorbar;
tk = linspace(0,1,2*length(a)+1);
%set(hh, 'YTick',tk(2:2:end),'YTickLabel', a','Position',[0.93 0.168 0.022 0.7]);
set(hh, 'YTick',tk(2:2:end),'YTickLabel', a','Position',[0.93 0.168 0.022 0.7]);
hh.Label.String = 'Iteration number in the outer circle';
pos=get(hh,'Position');
pos(1)=pos(1)-0.035;
set(hh,'Position',pos);
end 

%% Figure of the goodness values of the outputs (g_y)
for i=1:size(g_y_norm{1},1)
    for j=1:size(g_y_norm{1},2)
        for k=1:size(g_y_norm,2)
        g_y_norm_abr{i,j}(k)=g_y_norm{k}(i,j);
        end
    end
end

C = {'b','r','g','m'}
CC={ '-','--','-.'}

figure
subplot(3,1,1:2)
h=zeros(5,1)
for j=1:nE
hhhh(j)=plot(0:iter-1,[g_y_norm_abr{i,j}],CC{j},'color','k')
hold on
end
hhhh(4)=plot(nan,nan,'square','color','b','MarkerFacecolor','b')
hhhh(5)=plot(nan,nan,'square','color','r','MarkerFacecolor','r')
hhhh(6)=plot(nan,nan,'square','color','g','MarkerFacecolor','g')
hhhh(7)=plot(nan,nan,'square','color','m','MarkerFacecolor','m')

for i=1:size(g_y_norm{1},1)
for j=1:nE
plot(0:iter-1,[g_y_norm_abr{i,j}],CC{j} ,'color',C{i})
hold on
end
end
legend(hhhh,'Expert 1.','Expert 2.','Expert 3.','y_1','y_2','y_3','y_4')
xlabel('Iteration number in the outer circle [-]')

% axes1 = axes('Parent',fig1)
  %text('Parent',axes1(2),'Interpreter','latex','String','$$\sum_{i=1}^{2n} |k_i-k_j|$$')
ylabel('Normalized goodness values belonging to the outputs [-]')

subplot(3,1,3)
plot(jj)
xlabel('Iteration number in the outer circle [-]')
ylabel('Number of the inner iterations [-]')
xlim([0,size(g_y_norm,2)-1])


%% Figure of the goodness values of the parameter PET (g_1)
for i=1:size(g_1_norm{1},1)
    for j=1:size(g_1_norm{1},2)
        for k=1:size(g_1_norm,2)
        g_1_norm_abr{i,j}(k)=g_1_norm{k}(i,j);
        end
    end
end

figure
for i=1:size(g_1_norm{1},1)
for j=1:size(g_1_norm{1},2)
plot(0:iter-1,[1/2 g_1_norm_abr{i,j}])
hold on
end
end
xlabel('Iteration number in the outer circle')
ylabel('g_1')

%% Figure of the goodness values of the parameter LDPE (g_4)
for i=1:size(g_4_norm{1},1)
    for j=1:size(g_4_norm{1},2)
        for k=1:size(g_4_norm,2)
        g_4_norm_abr{i,j}(k)=g_4_norm{k}(i,j);
        end
    end
end

figure
for i=1:size(g_4_norm{1},1)
for j=1:size(g_4_norm{1},2)
plot(0:iter-1,[1/2 g_4_norm_abr{i,j}])
hold on
end
end
xlabel('Iteration number in the outer circle')
ylabel('g_4')


%% Figure of the goodness values of the parameter Iron (g_8)
for i=1:size(g_8_norm{1},1)
    for j=1:size(g_8_norm{1},2)
        for k=1:size(g_8_norm,2)
        g_8_norm_abr{i,j}(k)=g_8_norm{k}(i,j);
        end
    end
end

figure
for i=1:size(g_8_norm{1},1)
for j=1:size(g_8_norm{1},2)
plot(0:iter-1,[1/2 g_8_norm_abr{i,j}])
hold on
end
end
xlabel('Iteration number in the outer circle')
ylabel('g_8')

%% Figure of the goodness values of the parameter Al (g_9)
for i=1:size(g_9_norm{1},1)
    for j=1:size(g_9_norm{1},2)
        for k=1:size(g_9_norm,2)
        g_9_norm_abr{i,j}(k)=g_9_norm{k}(i,j);
        end
    end
end

figure
for i=1:size(g_9_norm{1},1)
for j=1:size(g_9_norm{1},2)
plot(0:iter-1,[1/2 g_9_norm_abr{i,j}])
hold on
end
end
xlabel('Iteration number of the outer circle')
ylabel('g_9')



%% Number of iterations in the inner circle
figure;
plot(jj)
xlabel('Iteration number of the outer circle')
ylabel('Number of the inner iterations')

%% Resampling_function
function [xk, wk, idx] = resample(xk, wk, resampling_strategy)
Ns = length(wk);  % Ns = number of particles
wk = wk./sum(wk); % normalize weight vector 
switch resampling_strategy
   case 'multinomial_resampling'
      with_replacement = true;
      idx = randsample(1:Ns, Ns, with_replacement, wk);

%{
      %THIS IS EQUIVALENT TO:
      edges = min([0 cumsum(wk)'],1); % protect against accumulated round-off
      edges(end) = 1;                 % get the upper edge exact
      % this works like the inverse of the empirical distribution and returns
      % the interval where the sample is to be found
      [~, idx] = histc(rand(Ns,1), edges);
%}

   case 'systematic_resampling'
      % this is performing latin hypercube sampling on wk
      edges = min([0 cumsum(wk)'],1); % protect against accumulated round-off
      edges(end) = 1;                 % get the upper edge exact
      u1 = rand/Ns;
      % this works like the inverse of the empirical distribution and returns
      % the interval where the sample is to be found
      [~, idx] = histc(u1:1/Ns:1, edges);
   % case 'regularized_pf'      TO BE IMPLEMENTED

   % case 'stratified_sampling' TO BE IMPLEMENTED
   % case 'residual_sampling'   TO BE IMPLEMENTED
   otherwise
      error('Resampling strategy not implemented')
end;
xk = xk(:,idx);                    % extract new particles
wk = repmat(1/Ns, 1, Ns);          % now all particles have the same weight
return;  % bye, bye!!!
end