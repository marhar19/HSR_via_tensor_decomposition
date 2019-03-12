function [ kR ] = khatri_rao( A,B )
%khatri-rao using the matlab function for kronecker
[~, F] = size(A);

kR = [];
for f=1:F
 kR = [kR kron(A(:,f),B(:,f))];
end

end

