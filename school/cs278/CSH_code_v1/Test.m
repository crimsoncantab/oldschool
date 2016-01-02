A = imread('Saba1.bmp');
[h w d] = size(A);

imshow(A);
handle = impoly;
position = wait(handle);
mask = createMask(handle);
B = A;
B(repmat(mask, [1,1,3])) = 0;

while any(any(mask))
    disp(sum(sum(mask)));
    CSH_ann = CSH_nn(A,B,8,5,1,0,mask);
    changed = zeros(h, w);
    for i = 1:h
        for j = 1:w
            if (mask(i,j) == 1)
                num_neigh = 0;
                a_pix = B(i, j, 1:3);
                a_i = CSH_ann(i,j,2);
                a_j = CSH_ann(i,j,1);
                if (i < h) && (mask(i+1,j) == 0)
                    a_pix = a_pix + A(min(a_i+1, h), a_j, 1:3);
                    num_neigh = num_neigh + 1;
                end
                if (i > 1) && (mask(i-1,j) == 0)
                    a_pix = a_pix + A(max(a_i-1, 1), a_j, 1:3);
                    num_neigh = num_neigh + 1;
                end
                if (j < w) && (mask(i,j+1) == 0)
                    a_pix = a_pix + A(a_i, min(a_j+1, w), 1:3);
                    num_neigh = num_neigh + 1;
                end
                if (j > 1) && (mask(i,j-1) == 0)
                    a_pix = a_pix + A(a_i, max(a_j-1, 1), 1:3);
                    num_neigh = num_neigh + 1;
                end
                if (num_neigh ~= 0)
                    changed(i,j) = 1;
                    B(i,j, 1:3) = a_pix / num_neigh;
                end
            end
        end
    end
    
    for i = 1:h
        for j = 1:w
            if changed(i,j) == 1
                mask(i,j) = 0;
            end
        end
    end
    
end
imshow(B);
% disp(CSH_ann(1,1,1));
