%START: OWN CODE
clc
clear all

numTrials = 100;
results = zeros(1,numTrials);
%numAxis = 15;
for numAxes = 1:48
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
    
    %Apply PCA for dimension reduction on training data
    covariance = cov(train_data); %covariance of training data
    [P eigVal] = eig(covariance); %P: transformation matrix, eigval: diagonal matrix with eignevalues
    eigVal = diag(eigVal); %create a vector of eigenvalues
    [eigVal, rindices] = sort(-1*eigVal); %sort eigenvalues in descending order
    P = P(:,rindices); %sort the transformation matrix by descedning eigenvalues
    train_data = (P(:,1:numAxes)' * train_data')'; %Apply the first 3 transformation columns
    test_data = (P(:,1:numAxes)' * test_data')';
    %Use the training data to apply MLE
    %calculate the means
    means = zeros(2,numAxes);
    for i = 1:numAxes
        for j = 0:1
            means(i,j+1) = mean(train_data(train_data_class_variable==j,i));
        end
    end
    
    %calculate covariance matrix, its inverse and its determinant
    covariance1 = cov(train_data(train_data_class_variable==1,:));
    covariance0 = cov(train_data(train_data_class_variable==0,:));
    inver0 = inv(covariance0);
    inver1 = inv(covariance1);
    deter0 = det(covariance0);
    deter1 = det(covariance1);
    
    %calculate priors
    prior0tmp = length(train_data(train_data_class_variable==0));
    prior1tmp = length(train_data(train_data_class_variable==1));
    prior0 = prior0tmp/(prior0tmp+prior1tmp);
    prior1 = prior1tmp/(prior0tmp+prior1tmp);
    
    correct = 0;
    wrong = 0;
    for i = 1:length(test_data)
        x_minus_mean0 = test_data(i,:)' - means(:,1);
        x_minus_mean1 = test_data(i,:)' - means(:,2);
        likelihood0 = exp(-(x_minus_mean0'*inver0*x_minus_mean0)/2)/sqrt(deter0);
        likelihood1 = exp(-(x_minus_mean1'*inver1*x_minus_mean1)/2)/sqrt(deter1);
        post0 = prior0*likelihood0;
        post1 = prior1*likelihood1;
        if(post0 > post1 && test_data_class_variable(i) == 0)
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
disp("implementing PCA with " + numAxes + " principal components and MLE over " + numTrials + " trials, mean = " + mean(results) + "% and standard deviation = " + std(results));
%disp(numAxes)
%disp(mean(results))
%disp(std(results))
end

%END: OWN CODE