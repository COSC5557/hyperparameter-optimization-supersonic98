% definitions
nx = 500;
ny = 500; 
pore = ones(nx, ny);

n_circles = 20; % number of circles
max_rad = 40;   % maximum radii of circles
min_rad = 10;   % minimum radii of circles
r = randi([min_rad, max_rad], 1, n_circles);    % random radii in given range

buffer = 10;    % buffer layer thickness

% create linear indices
a = max_rad + 10;   
b = ny - max_rad -9;
linear_ids = zeros(size(a:b, 2)^2, 1);
count = 1;
for i = a:b
    for j = a:b
        id = sub2ind(size(pore), i, j);
        linear_ids(count) = id;
        count = count + 1;
    end
end

% create domain
for c = 1:n_circles
    % pick a random point
    idr = randi([1, length(linear_ids)], 1, 1);
    [ic, jc] = ind2sub(size(pore), linear_ids(idr));
    
    % add obstacle (circle)
    for i = 1:ny
        for j = 1:nx
            if sqrt((i-ic)^2 +(j - jc)^2) <=r(c)
                pore(i, j) = 0;
            end
        end
    end

    % remove cells belonging to and surrounding the obstacle to make sure they don't touch
    ids_to_remove = zeros(ny*nx, 1);
    count = 1;
    for i = 1:ny
        for j = 1:nx
            if sqrt((i-ic)^2 +(j - jc)^2) <=max(r)*2
                id = sub2ind(size(pore), i, j);
                ids_to_remove(count) = id;
                count = count + 1;
            end
        end
    end
    
    % shrink linear index array
    ids_to_remove = ids_to_remove(ids_to_remove~=0);
    linear_ids = linear_ids(~ismember(linear_ids, ids_to_remove));

    imagesc(pore)
    colormap gray
    drawnow
end

% add buffer
pore(2:end-1, 1:buffer) = 1;
pore(2:end-1, ny-buffer+1:ny) = 1;

% add upper and lower boundaries
pore(1, :) = 0;
pore(end, :) = 0;

