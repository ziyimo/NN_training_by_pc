%% compare training data
ind = randi(size(X, 1));

figure;
colormap(gray);
image(reshape(X(ind, :), [28, 28])*255);

figure;
colormap(gray);
image(reshape(lrn_img(:, ind), [20, 20])*255);

%% Save data in desired format

save('mnist_2020_train.mat', 'X', 'y');

%% compare testing data
ind = randi(size(test_X, 1));

figure;
colormap(gray);
image(reshape(test_X(ind, :), [28, 28])*255);

figure;
colormap(gray);
image(reshape(tst_img(:, ind), [20, 20])*255);

%% Save data in desired format

save('mnist_2020_test.mat', 'test_X', 'test_Y');

%% Visualize data
ind = randi(size(X, 1));

figure;
colormap(gray);
image(reshape(X(ind, :), [20, 20])*255);
y(ind)