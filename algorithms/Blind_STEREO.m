function [ A_hat,B_hat,C,cost ] = Blind_STEREO( H,M,P3,MAXIT,A_hat,B_hat,A_tilde,B_tilde,C_tilde,lamda )
% Blind STEREO algorithm
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

[Ih,Jh,Kh]=size(H);
[I,J,Km]=size(M);
%% create the matrix equivalent models for the input tensor
% H1=zeros(Kh*Jh,Ih);
% H2=zeros(Kh*Ih,Jh);
% H3=zeros(Ih*Jh,Kh);

H1=reshape(H,[Ih,Jh*Kh])';

H2=reshape(permute(H,[2 1 3]),[Jh,Kh*Ih])';

H3=reshape(H,[Ih*Jh,Kh]);

nH=norm(H3,'fro');

M1=reshape(M,[I,J*Km])';

M2=reshape(permute(M,[2 1 3]),[J,Km*I])';

M3=reshape(M,[I*J,Km]);


nM=norm(M3,'fro');
%% initialize
temp_tilde=khatri_rao(B_tilde,A_tilde);
C=(temp_tilde\H3)';


%% Alternating least squares
eps=1e-03;
cost(1)=inf;
temp3m=khatri_rao(B_hat,A_hat);
Km= (B_hat'*B_hat).*(A_hat'*A_hat);
inv_K=pinv(Km);
M_ferror=norm(M3-temp3m*C_tilde','fro');

for iter=2:MAXIT
    %     iter
    temp1h=khatri_rao(C,B_tilde);
    Kh= (C'*C).*(B_tilde'*B_tilde);
    A_tilde=(Kh\(temp1h'*H1))';
    
    temp2h=khatri_rao(C,A_tilde);
    Kh= (C'*C).*(A_tilde'*A_tilde);
    B_tilde=(Kh\(temp2h'*H2))';
    
    
    temp3h=khatri_rao(B_tilde,A_tilde);
    Kh=lamda*(B_tilde'*B_tilde).*(A_tilde'*A_tilde);
    As=P3'*P3;
    Bs=Kh*inv_K;
    Cs=(lamda*H3'*temp3h+P3'*M3'*temp3m)*inv_K;
    C=sylvester(full(As),Bs,Cs);
    C_tilde=P3*C;
    
%         S1_hat1=khatri_rao(C,B_hat)*A_hat';
%     norm(S1-S1_hat1,'fro')/norm(S1,'fro')
    
    
    temp1m=khatri_rao(C_tilde,B_hat);
    Kh= (C_tilde'*C_tilde).*(B_hat'*B_hat);
    A_hat=(Kh\(temp1m'*M1))';
    
    temp2m=khatri_rao(C_tilde,A_hat);
    Kh= (C_tilde'*C_tilde).*(A_hat'*A_hat);
    B_hat=(Kh\(temp2m'*M2))';
    
    
    temp3m=khatri_rao(B_hat,A_hat);
    Km= (B_hat'*B_hat).*(A_hat'*A_hat);
    inv_K=pinv(Km);
    temp3h=khatri_rao(B_tilde,A_tilde);
    Kh=lamda*(B_tilde'*B_tilde).*(A_tilde'*A_tilde);
    As=P3'*P3;
    Bs=Kh*inv_K;
    Cs=(lamda*H3'*temp3h+P3'*M3'*temp3m)*inv_K;
    C=sylvester(full(As),Bs,Cs);
    C_tilde=P3*C;
    %     %     C_hat=P3*C;
    %         C1=(temp3h\H3t)';
%     S1_hat1=khatri_rao(C,B_hat)*A_hat';
%     norm(S1-S1_hat1,'fro')/norm(S1,'fro')
    %     S1_hat1=khatri_rao(V*C,B)*A';
    %     norm(S1-S1_hat1,'fro')/norm(S1,'fro')
    %     H_ferror=norm(H3-temp3h*C','fro');
    %     cost(iter)=H_ferror+M_ferror;
    %     %     drawnow;semilogy(cost);
    %     if (H_ferror/nH)<eps && (M_ferror/nM)<eps
    %         break
    %     end
end

end
