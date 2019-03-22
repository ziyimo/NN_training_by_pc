%% Load data
clear ; close all; clc;

dfile = '0306_112005.mat'; % MNIST5000
load('mnist_5000.mat');
load(dfile);

%% Plot output channels
MAX_IT = 2000;
%ind = randi(size(X, 1));

% DEMO INDICES
ind = 2461;

% MISMATCH INDICES
%ind = 4892; % 7 & 9
%ind = 4775; % 9 & 0
%ind = 2167;
%ind = 4326;


params.int_step = 0.1;
[pred_y, out, its] = predict_dynam(X(ind, :)', w_pc, b_pc, params, 100, 0, MAX_IT, 1);

fprintf('DIGIT: %d\n', y(ind));
fprintf('FFD prediction: %d\n', predict(X(ind, :)', w_pc, b_pc, params));

s = stackedplot(out', 'LineWidth', 1.5);
nPlots = size(out, 1);
for pt = 1:nPlots
%     subplot(nPlots, 1, pt);
%     plot(out(pt, :));
    s.AxesProperties(pt).YLimits = [0 1];
    s.DisplayLabels{pt} = int2str(pt);
end
s.FontSize = 15;

figure;
colormap(gray);
image(reshape(X(ind, :), [20, 20])*255);

%% Test set accuracy using predict_dynam()

%idx = randsample(size(X, 1),500);
idx = (1:size(X, 1))'; % all samples
MAX_IT = 2000;

y_dynmpred = zeros(size(idx));
conv_its = zeros(size(idx));

params.int_step = 0.2;
for ii=1:size(idx, 1)
    [y_dynmpred(ii), ~, conv_its(ii)] = predict_dynam(X(idx(ii), :)', w_pc, b_pc, params, 100, 0.9, MAX_IT, 1);
    fprintf('\nSample %d, iterations %d\n', ii, conv_its(ii));
end

fprintf('\nTraining Set Accuracy (feedfwd): %f\n', mean(double(y_pred(idx) == y(idx)')) * 100);
fprintf('\nTraining Set Accuracy (dynm): %f\n', mean(double(y_dynmpred == y(idx))) * 100);
fprintf('\nAgreement b/n feedfwd and dynm: %f\n', mean(double(y_dynmpred == y_pred(idx)')) * 100);

mismatch = [idx(y_dynmpred ~= y_pred(idx)'), y_dynmpred(y_dynmpred ~= y_pred(idx)'), y_pred(idx(y_dynmpred ~= y_pred(idx)'))'];

%histogram(conv_its);
%%
% idx_conv = idx(conv_its ~= MAX_IT);
% fprintf('CONVERGED PREDICTION ONLY\n');
% fprintf('\nTraining Set Accuracy (feedfwd): %f\n', mean(double(y_pred(idx_conv) == y(idx_conv)')) * 100);
% fprintf('\nTraining Set Accuracy (dynm): %f\n', mean(double(y_dynmpred(~(conv_its == MAX_IT)) == y(idx_conv))) * 100);
% fprintf('\nAgreement b/n feedfwd and dynm: %f\n', mean(double(y_dynmpred(~(conv_its == MAX_IT)) == y_pred(idx_conv)')) * 100);
