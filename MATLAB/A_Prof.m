fid = fopen('A_prof.txt', 'r');
G = fscanf(fid, '%g');
fclose(fid);
k=1;
A(1:21,3)=0;
for i=1:length(G)/3
    B(i,1)=G(k);
    B(i,2)=G(k+1);
    B(i,3)=G(k+2);
    k=k+3;
end
x1=0:1/(length(B(:,1))-1):1;
x11=0:1/20:1;
X=B(:,1);
Ycr=B(:,2);
Ysim=B(:,3);
yi1(:,1) = interp1(x1, X, x11);
yi2(:,1) = interp1(x1, Ycr, x11);
yi3(:,1) = interp1(x1, Ysim, x11);
A(:,1)=yi1;
A(:,2)=yi2;
A(:,3)=yi3;
clear ans fid k G i B X Yc Ysim x1 x11 yi1 yi2 yi3 Ycr
X=A(:,1);
Ycr=A(:,2);
Ysim=A(:,3);
if exist('f_0','var') && f_0~=0
    Yv = Ycr * f_0;
elseif exist('f_k','var')
    Yv = Ycr * f_k;
else
    error('f_0 или f_k не определены');
end

% --- Ќќ¬јя √≈Ќ≈–ј÷»я Ќ≈–ј¬Ќќћ≈–Ќќ… —≈“ » ѕќ ’ќ–ƒ≈ ---
% „исло панелей по хорде (остаЄтс€ 1/raz_chord)
n_chord = round(1 / raz_chord);
% —оздание сгущЄнной сетки: xi = (i/n_chord)^2, i=0..n_chord
% ѕерва€ точка = 0 (передн€€ кромка), последн€€ = 1 (задн€€ кромка)
xi = linspace(0, 1, n_chord+1).^2;
% ----------------------------------------------------

% »нтерпол€ци€ координат профил€ на новую сетку
yi = interp1(X, Yv, xi);
Nod(:,1) = xi;
Nod(:,2) = yi;
Nod(:,3) = 0;

% —ортировка по убыванию x (задн€€ кромка -> передн€€)
for i=1:length(Nod)
    for j=2:length(Nod)
        if Nod(j-1,1) < Nod(j,1)
            x = Nod(j-1,1);
            y = Nod(j-1,2);
            z = Nod(j-1,3);
            Nod(j-1,1) = Nod(j,1);
            Nod(j-1,2) = Nod(j,2);
            Nod(j-1,3) = Nod(j,3);
            Nod(j,1) = x;
            Nod(j,2) = y;
            Nod(j,3) = z;
        end
    end
end
clear X Ycr Yn Ysim Yv a i xi yi yi1 xi1 x y z j