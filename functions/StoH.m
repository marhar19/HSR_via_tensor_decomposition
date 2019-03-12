function [HSI,P1,P2,P_H,BluKer]=StoH(SRI,ratio,kernel_length,sig,start_pos,type)
% Generate an HSI fro an SRI
% Charilaos Kanatsoulis, UMN, January 7, 2018
[I,J,K]=size(SRI);
if strcmp(type,'gaussian')
BluKer = fspecial('gaussian',[kernel_length kernel_length],sig);
end
if strcmp(type,'average')
BluKer = fspecial('average',[kernel_length kernel_length]);
end
veck=sqrt(diag(BluKer));
PP1=toeplitz([veck(ceil(kernel_length/2):kernel_length); zeros(I-kernel_length,1);veck(1:floor(kernel_length/2))], [veck(ceil(kernel_length/2):kernel_length); zeros(I-kernel_length,1);veck(1:floor(kernel_length/2))]);
P1=PP1(start_pos(1):ratio:end,:);
PP2=toeplitz([veck(ceil(kernel_length/2):kernel_length); zeros(J-kernel_length,1);veck(1:floor(kernel_length/2))], [veck(ceil(kernel_length/2):kernel_length); zeros(J-kernel_length,1);veck(1:floor(kernel_length/2))]);
P2=PP2(start_pos(2):ratio:end,:);
P1=sparse(P1);
P2=sparse(P2);
P_H=kron(P2,P1);
for k=1:K
    HSI(:,:,k)=P1*SRI(:,:,k)*P2';
end
end
