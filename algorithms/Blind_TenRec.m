function[A_hat,B_hat,C_hat,A_tilde,B_tilde,C_tilde]=Blind_TenRec(MSI,H3,maxit,t_rank)
% Blind Ten-Rec algorithm
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

[IJ_h,~]=size(H3);
[I,J,~]=size(MSI);
ratio=sqrt(I*J/IJ_h);
Ih=I/ratio;
Jh=J/ratio;

[U,~]=cpd(MSI,t_rank,'MaxIter',maxit);
A_hat=U{1};
B_hat=U{2};
C_tilde=U{3};
% temp=khatri_rao(B_hat,A_hat);

for i = 1:Ih
    A_tilde(i,:) = 1/ratio*sum(A_hat((i-1)*ratio+1:i*ratio,:));
end
for j = 1:Jh
    B_tilde(j,:) = 1/ratio*sum(B_hat((j-1)*ratio+1:j*ratio,:));
end
temp_tilde=khatri_rao(B_tilde,A_tilde);
C_hat=(temp_tilde\H3)';
end
