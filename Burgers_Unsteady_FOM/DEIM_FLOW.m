clc;
clear variables;
load 'Results.mat';
Data = u_Mat;
[s1,s2] = size(Data);
nxf = s1;
[phin,Svn,Vn] = svd(Data);
svn_max = max(diag(Svn));
Svn_normalized = diag(Svn)./svn_max;
count = 5;
phin = phin(:,1:count);
IM = eye(nxf,nxf);
count_d = zeros(count,1);
r = zeros(nxf,1);
P = zeros(nxf,1);
Psi_diem = phin;
U_psi = zeros(nxf,1);
k =0;
j =1;
for i = 1:nxf
r(i) = Psi_diem(i,1);
if abs(r(i)) > k
    k=abs(r(i));
    count_d(j) = i;
end
end
U_psi(:,1)=Psi_diem(:,1);
P(:,1)=IM(:,count_d(j));
for j = 2:count
k=0;
 A_psi = transpose(P)*U_psi;
 B_psi =  transpose(P)*Psi_diem(:,j);
 c = A_psi\B_psi;
 r = abs(Psi_diem(:,j)-U_psi*c);
 for i =1:nxf
 if abs(r(i)) > k
    k=abs(r(i));
    count_d(j) =i;
end
end
U_psi(:,1:j)=Psi_diem(:,1:j);
P(:,1:j) = [P(:,1:j-1) IM(:,count_d(j))]; 
j
end
% clearvars -except U_psi count_d P
rahula =inv(transpose(P)*U_psi);
rahulb = inv(U_psi(count_d,:));
Test_interpol = U_psi*(rahula);
count_d_Test = count_d;
Data_set_Diff_Test= Data(count_d_Test,:);
clearvars -except Test_interpol count_d_Test Data_set_Diff_Test Data phin
Mod_Test = Test_interpol*Data_set_Diff_Test;
% plot(Mod_Test(:,20));
% hold on;
% plot(Data(:,20),'--r*');
















