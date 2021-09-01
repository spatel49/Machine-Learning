%START: OWN CODE
clc
clear all
close all

numTrials = 100;
results = zeros(1,numTrials);

for trial = 1:numTrials
    data = readmatrix('aggregate16.csv'); %load the data
    
    %randomize the data
    rp = randperm(length(data));
    data = data(rp,:);
    %split the data in half into training and testing datasets
    split = length(data)/2;
    train_data = data(1:split,:);
    test_data = data(split+1:end,:);
    
    %separate class variable (feature 9) from other features
    train_data_class_variable = (train_data(:,49));
    train_data = (train_data(:,1:48));
    test_data_class_variable = (test_data(:,49));
    test_data = (test_data(:,1:48));
    
    train_data_0 = [];
    train_data_1 = [];
    %split the training set into 2 classes
    for i = 1:length(train_data)
        if (train_data_class_variable(i) == 0)
            train_data_0 = [train_data_0; train_data(i,:)];
        elseif (train_data_class_variable(i) == 1)
            train_data_1 = [train_data_1; train_data(i,:)];
        end
    end
    
    %class mean
    mean_0 = mean(train_data_0);
    mean_1 = mean(train_data_1);
    %within-class scatter matrices
    S0 = cov(train_data_0)*(length(train_data_0) - 1);
    S1 = cov(train_data_1)*(length(train_data_1) - 1);
    Sw = S0 + S1;
    Swinv = inv(Sw);
    %optimal line direction
    v = Swinv * (mean_0 - mean_1)';
    %projecting the training and testing data to the optimal line direction
    train_data_0_proj = train_data_0 * v;
    train_data_1_proj = train_data_1 * v;
    test_data_proj = test_data * v;
    %plotting the projected data on a linear scale
    if (trial == 1)
        plot(train_data_0_proj,0,"bX")
        hold on
        plot(train_data_1_proj,0,"rX")
        hold off
    end
    mean0 = mean(train_data_0_proj);
    mean1 = mean(train_data_1_proj);
    var0 = var(train_data_0_proj);
    var1 = var(train_data_1_proj);
    prior0tmp = length(train_data_0_proj);
    prior1tmp = length(train_data_1_proj);
    prior0 = prior0tmp/(prior0tmp+prior1tmp);
    prior1 = prior1tmp/(prior0tmp+prior1tmp);
    
    correct = 0;
    wrong = 0;
    for i = 1:length(test_data_proj)
        lklhood0 = exp(-(test_data_proj(i)-mean0)^2/(2*var0)) /sqrt(var0);
        lklhood1 = exp(-(test_data_proj(i)-mean1)^2/(2*var1)) /sqrt(var1);
        post0 = lklhood0 * prior0;
        post1 = lklhood1 * prior1;
        if (post0 > post1 && test_data_class_variable(i) == 0)
            correct = correct+1;
        elseif(post0 < post1 && test_data_class_variable(i) == 1)
            correct = correct+1;
        else
            wrong = wrong+1;
        end
    end
    accuracy = correct * 100 / (correct + wrong);
    results(trial) = accuracy;
end
disp("implementing MLE over " + numTrials + " trials, mean = " + mean(results) + "% and standard deviation = " + std(results));

%END: OWN CODE