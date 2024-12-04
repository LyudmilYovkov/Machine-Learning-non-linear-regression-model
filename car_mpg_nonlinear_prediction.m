clear; clc;

% load the data from a *.csv file
filename = "auto-mpg.csv";
csv_data = readtable(filename, "VariableNamingRule", "preserve");

% data table
data_table = table(csv_data.acceleration, ...
                   csv_data.displacement, ...
                   csv_data.horsepower, ...
                   csv_data.mpg, ...
                   csv_data.weight);
% rename the columns
data_table = renamevars(data_table, ...
        ["Var1", "Var2", "Var3", "Var4", "Var5"], ...
        ["Acceleration", "Displacement", "Horsepower", "MPG", "Weight"]);

% remove missing values
data_table = rmmissing(data_table);

% train-test split: 70% train data, 30% test data
cv = cvpartition(size(data_table,1), "HoldOut", 0.3);
idx = cv.test;
train_data = data_table(~idx,:);
test_data = data_table(idx,:);

% predictors
x_train = [train_data.Horsepower, train_data.Weight];

% response
y_train = train_data.MPG;

% non-linear regression model
model_fun = @(w,x) w(1)*x(:,1) + w(2)*x(:,2) + w(3).*x(:,1).*x(:,2) + w(4);
%linear_model = fitlm(x_train, y_train);
w_init = [-1, -0.5, 0.2, 50];
nonlin_model = fitnlm(x_train, y_train, model_fun, w_init);
coeffs = nonlin_model.Coefficients.Estimate;
disp("fitnlm() results: ")
disp(nonlin_model)

% test the model
x_test = [test_data.Horsepower, test_data.Weight];
y_test = test_data.MPG;
y_pred = predict(nonlin_model, x_test);
abs_err = abs(y_pred - y_test);
eval_table = table(y_pred, y_test, abs_err);
disp(eval_table)

%---------------------------------------------------------% 
% predictions on new data                                 %
% middle car size: weight ~ 3300 pounds, 180 < hp < 200   %
% large car size : weight ~ 4400 pounds, 200 < hp < 300   %
%---------------------------------------------------------% 
weight = 3200; hp = 120;
mpg = predict(nonlin_model, [hp, weight]);
disp(['middle car size: ', num2str(mpg)])
% % %
weight = 4000; hp = 250;
mpg = predict(nonlin_model, [hp, weight]);
disp(['large car size: ', num2str(mpg)])

%-------------------------%
% training results plot   %
%-------------------------%
hp_vec = linspace(min(x_train(:,1)), max(x_train(:,1)), 30);
w_vec = linspace(min(x_train(:,2)), max(x_train(:,2)), 30);
[w_i,hp_i] = meshgrid(w_vec, hp_vec);
%mpg_i = coeffs(1) + coeffs(2) * hp_i + coeffs(3) * w_i;
mpg_i = coeffs(1) * hp_i + ...
        coeffs(2) * w_i + ...
        coeffs(3) .* hp_i .* w_i + ...
        coeffs(4);

figure(1);
% % %
subplot(1,2,1)
plot3(x_train(:,1), x_train(:,2), y_train, 'mo', 'MarkerFaceColor', 'magenta')
hold on, grid on 
set(gca, 'FontSize', 12)
xlabel('\bf{Horsepower}')
ylabel('\bf{Weight}')
zlabel('\bf{MPG}')
colormap winter
title('\bf{Measurement: training data}')
% % %
subplot(1,2,2)
surf(hp_i, w_i, mpg_i)
hold on, grid on
set(gca, 'FontSize', 12)
xlabel('\bf{Horsepower}')
ylabel('\bf{Weight}')
zlabel('\bf{MPG}')
colormap winter
plot3(x_train(:,1), x_train(:,2), y_train, 'mo', 'MarkerFaceColor', 'magenta')
title('$ \hat{y}(x_{1},x_{2}) = b + w_{1}x_{1} + w_{2}x_{2} + w_{3}x_{1}x_{2}$', ...
      'interpreter', 'latex')

%---------------------%
% test results plot   %
%---------------------%
hp_vec = linspace(min(x_test(:,1)), max(x_test(:,1)), 30);
w_vec = linspace(min(x_test(:,2)), max(x_test(:,2)), 30);
[w_i,hp_i] = meshgrid(w_vec, hp_vec);
mpg_i = coeffs(1) * hp_i + ...
        coeffs(2) * w_i + ...
        coeffs(3) .* hp_i .* w_i + ...
        coeffs(4);

figure(2);
% % %
subplot(1,2,1)
plot3(x_test(:,1), x_test(:,2), y_test, 'bo', 'MarkerFaceColor', 'cyan')
hold on, grid on 
set(gca, 'FontSize', 12)
xlabel('\bf{Horsepower}')
ylabel('\bf{Weight}')
zlabel('\bf{MPG}')
colormap winter
title('\bf{Measurement: test data}')
% % %
subplot(1,2,2)
surf(hp_i, w_i, mpg_i)
hold on, grid on
set(gca, 'FontSize', 12)
xlabel('\bf{Horsepower}')
ylabel('\bf{Weight}')
zlabel('\bf{MPG}')
colormap winter
plot3(x_test(:,1), x_test(:,2), y_test, 'bo', 'MarkerFaceColor', 'cyan')
title('$ \hat{y}(x_{1},x_{2}) = b + w_{1}x_{1} + w_{2}x_{2} + w_{3}x_{1}x_{2}$', ...
      'interpreter', 'latex')

%--------------%
% error plot   %
%--------------%
figure(3)
% % %
subplot(2,2,1)
%plot(x_test(:,1), abs_err, 'ro')
plot(x_test(:,1), y_test, 'bo')
hold on, grid on
plot(x_test(:,1), y_pred, 'ro')
set(gca, 'FontSize', 14)
xlabel('\bf{Horsepower}')
ylabel('\bf{MPG}')
legend('\it{MPG test}', '\it{MPG predicted}')
% % %
subplot(2,2,2)
plot(x_test(:,2), y_test, 'bo')
hold on, grid on
plot(x_test(:,2), y_pred, 'ro')
set(gca, 'FontSize', 14)
xlabel('\bf{Weight}')
ylabel('\bf{MPG}')
legend('\it{MPG test}', '\it{MPG predicted}')
% % %
subplot(2,2,3)
plot(x_test(:,1), abs_err, 'mx')
hold on, grid on
set(gca, 'FontSize', 14)
xlabel('\bf{Horsepower}')
ylabel('\bf{Error}')
legend('\it{|MPG_{test} - MPG_{predicted}|}')
% % %
subplot(2,2,4)
plot(x_test(:,2), abs_err, 'mx')
hold on, grid on
set(gca, 'FontSize', 14)
xlabel('\bf{Weight}')
ylabel('\bf{Error}')
legend('\it{|MPG_{test} - MPG_{predicted}|}')