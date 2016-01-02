% images = {'pflower', 'signal'};
images = {'pflower'};
for i = 1:size(images, 2)
    Im = imread(strcat(images{i}, '.jpg'));
%     for lambda = 0.01:0.01:0.1
    for lambda = 0.01:0.01
%         S = L0SmoothingL(Im,lambda);
%         imwrite(S,strcat(images{i}, '-laplacian-', sprintf('%.2f', lambda), '.jpg'),'jpg');
%         S = L0Smoothing(Im,lambda);
%         imwrite(S,strcat(images{i}, '-xu-', sprintf('%.2f', lambda), '.jpg'),'jpg');
        S = L0Smoothing(Im,lambda);
        imshow(S);
%         imwrite(S,strcat(images{i}, '-hybrid-', sprintf('%.2f', lambda), '.jpg'),'jpg');
    end
end
