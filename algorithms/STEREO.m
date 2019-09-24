function [ A,B,C,cost ] = STEREO( H,M,P1,P2,P3,MAXIT,lamda,A,B,C,C_tilde )
%STEREO algorithm
% (c) Charilaos I. Kanatsoulis, University of Minnesota, Jan 7 , 2018
% kanat003@umn.edu
% 
% Reference 1: C.I. Kanatsoulis, X. Fu, N.D. Sidiropoulos and W.K. Ma, 
%``Hyperspectral Super-resolution: A Coupled Tensor Factorization
%Approach,'' IEEE Transactions in Signal Processing, 2018

% Reference 2: C.I. Kanatsoulis, X. Fu, N.D. Sidiropoulos and W.K. Ma, 
%``Hyperspectral Super-resolution via Coupled Tensor Factorization:
%Identifiability and Algorithms,'' IEEE International Conference on 
%Acoustics, Speech and Signal Processing (ICASSP), 2018
[Ih,Jh,K]=size(H);
[I,J,Km]=size(M);
%% create the matrix equivalent models for the input tensor
H1=reshape(H,[Ih,Jh*K])';
H2=reshape(permute(H,[2 1 3]),[Jh,K*Ih])';
H3=reshape(H,[Ih*Jh,K]);
nH=norm(H3,'fro');


M1=reshape(M,[I,J*Km])';
M2=reshape(permute(M,[2 1 3]),[J,Km*I])';
M3=reshape(M,[I*J,Km]);
nM=norm(M3,'fro');

%% initialize
B_tilde=P2*B;

%% Alternating Optimization
eps=1e-03;
cost(1)=inf;
for iter=2:MAXIT
    
    temp1h=khatri_rao(C,B_tilde);
    temp1m=khatri_rao(C_tilde,B);
    Kp=(C_tilde'*C_tilde).*(B'*B);
    K= (C'*C).*(B_tilde'*B_tilde);
    inv_K=pinv(K);
    As=lamda*(P1'*P1);
    Bs=Kp*inv_K;
    Cs=(lamda*P1'*H1'*temp1h+M1'*temp1m)*inv_K;
    A=sylvester(full(As),Bs,Cs);
    A_tilde=P1*A;
    
    temp2h=khatri_rao(C,A_tilde);
    temp2m=khatri_rao(C_tilde,A);
    Kp=(C_tilde'*C_tilde).*(A'*A);
    K= (C'*C).*(A_tilde'*A_tilde);
    inv_K=pinv(K);
    As=lamda*(P2'*P2);
    Bs=Kp*inv_K;
    Cs=(lamda*P2'*H2'*temp2h+M2'*temp2m)*inv_K;
    B=sylvester(full(As),Bs,Cs);
    B_tilde=P2*B;
     
    temp3h=khatri_rao(B_tilde,A_tilde);
    temp3m=khatri_rao(B,A);
    Kp=lamda*(B_tilde'*B_tilde).*(A_tilde'*A_tilde);
    K= (B'*B).*(A'*A);
    inv_K=pinv(K);
    As=P3'*P3;
    Bs=Kp*inv_K;
    Cs=(lamda*H3'*temp3h+P3'*M3'*temp3m)*inv_K;
    C=sylvester(full(As),Bs,Cs);
    C_tilde=P3*C;
     
%     H_ferror=norm(H3-temp3h*C','fro');
%     M_ferror=norm(M3-temp3m*C_tilde','fro');
%     cost(iter)=lamda*H_ferror+M_ferror;
%     if     abs((cost(iter)-cost(iter-1))/cost(iter-1))<eps
%         break
%     end
end

end

