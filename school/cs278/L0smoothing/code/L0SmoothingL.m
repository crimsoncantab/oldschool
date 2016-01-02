function S = L0SmoothingL(Im, lambda, kappa)


if ~exist('kappa','var')
    kappa = 2.0;
end
if ~exist('lambda','var')
    lambda = 2e-2;
end
S = im2double(Im);
betamax = 1e5;
% fl = [0,1,0;1,-4,1;0,1,0];
lapfilter = fspecial('laplacian', 0);
[N,M,D] = size(Im);
sizeI2D = [N,M];
otfFl = psf2otf(lapfilter,sizeI2D);
Numer1 = fft2(S);
Denom2 = abs(otfFl).^2;
if D>1
    Denom2 = repmat(Denom2,[1,1,D]);
end
beta = 2*lambda;
while beta < betamax
    Denom   = 1 + beta*Denom2;
    % lap subproblem
    lap = circfilter(S, lapfilter);
    if D==1
        t = (lap.^2)<lambda/beta;
    else
        t = sum((lap.^2),3)<lambda/beta;
        t = repmat(t,[1,1,D]);
    end
    lap(t)=0;
    % S subproblem
    Numer2 = circfilter(lap, lapfilter);
    FS = (Numer1 + beta*fft2(Numer2))./Denom;
    S = real(ifft2(FS));
    beta = beta*kappa;
    fprintf('.');
end
fprintf('\n');
end

function R = circfilter(M, H)
    pad = padarray(M, [1, 1], 'circular');
    for d = 1:size(M, 3)
        pad(:,:,d) = filter2(H, pad(:,:,d));
    end
    R = pad(2:end-1,2:end-1,:);
end