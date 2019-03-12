% Demo for Blind STEREO algorithm
% (c) Charilaos I. Kanatsoulis, University of Minnesota, Jan 7 , 2018
% nikos@umn.edu
% 
% Reference 1: C.I. Kanatsoulis, X. Fu, N.D. Sidiropoulos and W.K. Ma, 
%``Hyperspectral Super-resolution: A Coupled Tensor Factorization
%Approach,'' IEEE Transactions in Signal Processing, 2018

% Reference 2: C.I. Kanatsoulis, X. Fu, N.D. Sidiropoulos and W.K. Ma, 
%``Hyperspectral Super-resolution via Coupled Tensor Factorization:
%Identifiability and Algorithms,'' IEEE International Conference on 
%Acoustics, Speech and Signal Processing (ICASSP), 2018

clc; clear; close all;
% ============================================
load('PaviaU.mat');   
load('paviaU_spec.mat');
load('QuickBird_spec.mat');
SRI = paviaU(2:end-1,3:end-2,1:end);

[I, J, K] = size(SRI);
S1=reshape(SRI,[I,K*J])'; %mode 1 unfolding
S3=reshape(SRI,[I*J,K]);  %mode 3 unfolding
% ============================================
% --------generate MS image
PM = ConstructP_M(paviaU_spec(1:end),MS_spec);
PM=sparse(PM);
M3=S3*PM';
MSI = reshape(M3,I,J,[]);
K_M=length(MS_spec);
%% add noise
signal_powerM=norm(M3,'fro')^2/(I*J*K_M);
SNR_M=25;
noise_power=signal_powerM/10^(SNR_M/10);
noise_tenM=sqrt(noise_power)*randn(I,J,K_M);
noiseM=reshape(noise_tenM,[I*J*K_M,1]);
noise_powerM=norm(noiseM)^2/(I*J*K_M);
MSI=MSI+noise_tenM;
M3=reshape(MSI,[I*J,K_M]);
% 10*log10(signal_powerM/noise_powerM)
fprintf('MSI SNR = %2.1f \n',10*log10(signal_powerM/noise_powerM))

% ===========================================
%% constract HS image
ratio = 4;
Ih=I/ratio;
Jh=J/ratio;
global kernel_length
global sig
kernel_length=7;
sig = (1/(2*(2.7725887)/ratio^2))^0.5;
% sig=4
start_pos(1)=4; % The starting point of downsampling
start_pos(2)=4; % The starting point of downsampling


[HSI,P1,P2,P_H,BluKer]=StoH(SRI,ratio,kernel_length,sig,start_pos,'gaussian');

H3=reshape(HSI,[Ih*Jh,K]);

%% add noise
signal_powerH=norm(H3,'fro')^2/(Ih*Jh*K);
SNR_H=25;
noise_power=signal_powerH/10^(SNR_H/10);
noise_tenH=sqrt(noise_power)*randn(Ih,Jh,K);
noiseH=reshape(noise_tenH,[Ih*Jh*K,1]);
noise_powerH=norm(noiseH)^2/(Ih*Jh*K);
HSI=HSI+noise_tenH;
H3=reshape(HSI,[Ih*Jh,K]);
fprintf('HSI SNR = %2.1f \n',10*log10(signal_powerH/noise_powerH))

% 10*log10(signal_powerH/noise_powerH)
%% Blind TenRec (initialization algorithm)
t_rank=400; % rank of the tensor -- change for different noise levels
maxit=25; % number of CPD iterations

[A_hat,B_hat,C_hat,A_tilde,B_tilde,C_tilde]=Blind_TenRec(MSI,H3,maxit,t_rank);

S1_hat1=khatri_rao(C_hat,B_hat)*A_hat';

%% Blind STEREO
lamdab=1;

bs_iter=4; % blind stereo iterations -- play between 3-15
[ A_hatb,B_hatb,C_hatb,cost ] = Blind_STEREO( HSI,MSI,PM,bs_iter,A_hat,B_hat,A_tilde,B_tilde,C_tilde,lamdab);


S1_hat1=khatri_rao(C_hatb,B_hatb)*A_hatb';
nmse=norm(S1-S1_hat1,'fro')/norm(S1,'fro'); %NMSE
rsnr=10*log10(norm(S1,'fro')^2/norm(S1-S1_hat1,'fro')^2); %RSNR
fprintf('R-SNR= %3.4f, NMSE=%1.4f \n',rsnr,nmse)
SSI_hat=reshape(S1_hat1',[I J K]);
