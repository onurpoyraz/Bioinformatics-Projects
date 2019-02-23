% MULTILAYER PERCEPTRON MODEL
clear all;
clc;
%% Read the input and desired output from text for train and test structures
% This code can be read only ALL-AML.txt file with header of
formatSpec = '%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%C';
A=readtable('ALL-AML.txt','Delimiter',',','Format',formatSpec); %%This code can only read the iris.txt
X_all = A{1:72,{'input1','input2','input3','input4','input5','input6','input7','input8','input9','input10','input11','input12','input13','input14','input15','input16','input17','input18','input19','input20','input21','input22','input23','input24','input25','input26','input27','input28','input29','input30'}};
d_name=A{1:72,{'output'}};
for i=1:72
    if d_name(i)=='ALL'
        d_all(i,:)=[1 0];
    elseif d_name(i)=='AML'
        d_all(i,:)=[0 1];
    end
end
train=38; %%Define number of train sampler pattern
%indis=randperm(72);
indis = 1:1:72;
for i=1:72
    if i<=train
        X(i,:)=X_all(indis(i),:);
        d(i,:)=d_all(indis(i),:);
    else
        X_test(i-train,:)=X_all(indis(i),:);
        d_test(i-train,:)=d_all(indis(i),:);
        d_result(i-train,1)=d_name(indis(i));
    end
end
%% User defined parameters
lf=0.01; %% setting learning factor
moment=0;%% set momentum
nodes=[30 5 2]; %% Shape of structure the first term must be equal to #of inputs and second term must be equal to #of output
%% Calculating number of layer and max layer length
layer=length(nodes)-1;
l_max=0;
samplerpattern=length(X(:,1)); %% calculate the length of the sampler pattern
for i=1:length(nodes)
    if nodes(i)>=l_max
        l_max=nodes(i); %% calculate the longest layer in the structure
    end
end
%% Sigmoid function
sigmoid = @(x) tanh(x);
sigmoid_diff= @(x) sech(x);
% sigmoid = @(x) 1/(1+exp(-x));
% sigmoid_diff= @(x) (1/(1+exp(-x)))*(1-(1/(1+exp(-x))));
%% ?nitializing Weight and Teta Matrices
w=zeros(l_max,l_max,layer);
delta_w=zeros(l_max,l_max,layer);
delta_w_old=zeros(l_max,l_max,layer);
teta=zeros(layer,l_max);
delta_teta=zeros(layer,l_max);
delta_teta_old=zeros(layer,l_max);
for k=1:layer
    for j=1:nodes(k+1)
        for i=1:nodes(k)
            w(i,j,k)=(rand-0.5)/10;  %% initialize weights randomly
        end
    end
end
for k=1:layer-1
    for j=1:nodes(k+1)
        teta(k,j)=(rand-0.5)/10;     %%initialize threshold randomly for each neuron
    end
end
%% Implementation of Algorithm
epoch=0;
while(1)
    epoch=epoch+1; %counting number of epoch
    error=zeros(1,samplerpattern);
    output=zeros(length(d(:,1)),length(d(1,:)));
    index=randperm(samplerpattern); %in each epoch this parts makes sampler patterns comes randomly
    for sample=1:samplerpattern
        s=index(sample);
        y=zeros(layer,l_max);
        y_diff=zeros(layer,l_max);
        sigma=zeros(layer,l_max);
        x=zeros(layer,l_max);
        input=X(s,:);
        %% Calculate the Outputs and Find the Last Error at the System
        for k=1:layer
            for j=1:nodes(k+1)
                for i=1:nodes(k);
                    x(k,j)=x(k,j)+input(i)*w(i,j,k); %%forward propogation
                end
                if k==layer
                    y(k,j)=x(k,j);         %%find output of the structure
                    sigma(k,j)=d(s,j)-y(k,j);  %%find sigma for back propogation
                    output(s,j)=y(k,j);     
                    error(1,s)=error(1,s)+(sigma(k,j)^2); %%find error
                else
                    x(k,j)=x(k,j)+teta(k,j);     %%find hidden layer outputs with sigmoid calculate sigmoid_diff additionally
                    y(k,j)=sigmoid(x(k,j));
                    y_diff(k,j)=sigmoid_diff(x(k,j));
                end
                error(1,s)=error(1,s)/2;
            end
            if k~=layer
                input=y(k,:);
            end
        end
        %% Calculate Delta Weight and Delta Teta Matrices and Update Them
        for k=layer:-1:1
            for i=1:nodes(k)
                for j=1:nodes(k+1)
                    if k~=3
                        delta_teta(k,j)=lf*sigma(k,j)+moment*delta_teta_old(k,j);
                        teta(k,j)=teta(k,j)+delta_teta(k,j); %% update thresholds for each neuron
                    end
                    if k==1
                        delta_w(i,j,k)=lf*sigma(k,j)*X(s,i)+moment*delta_w_old(i,j,k);
                        w(i,j,k)=w(i,j,k)+delta_w(i,j,k);  %% update weights for input layer if layer=1
                    else
                        delta_w(i,j,k)=lf*sigma(k,j)*y(k-1,i)+moment*delta_w_old(i,j,k);
                        w(i,j,k)=w(i,j,k)+delta_w(i,j,k);  %% update weights for hidden and output layer
                        sigma(k-1,i)=sigma(k-1,i)+w(i,j,k)*sigma(k,j); %% derive new sigmas for the neurons which are at(k-1)th layer
                    end
                end
                if k~=1
                    sigma(k-1,i)=sigma(k-1,i)*y_diff(k-1,i);
                end
            end
        end
        %% For momentum term this part hides the old values of delta weight
        delta_w_old=delta_w;
        delta_teta_old=delta_teta;
    end
    %% Calculate total cost function and plot it
    err(epoch)=sum(error);
    figure(1);
    plot(err)
    if err(epoch)<0.1
        fprintf('converged at epoch: %d\n',epoch);
        break
    end
end
%% Test Structure
out=zeros(length(d_test(:,1)),length(d_test(1,:)));
for s=1:length(d_test(:,1))
    x=zeros(layer,l_max);
    y=zeros(layer,l_max);
    in=X_test(s,:);
    for k=1:layer
        for j=1:nodes(k+1)
            for i=1:nodes(k);
                x(k,j)=x(k,j)+in(i)*w(i,j,k);
            end
            if k==layer
                y(k,j)=x(k,j);
                out(s,j)=y(k,j);
            else
                x(k,j)=x(k,j)+teta(k,j);
                y(k,j)=sigmoid(x(k,j));
            end
        end
        if k~=layer
            in=y(k,:);
        end
    end
    if out(s,1)<out(s,2)
        actual(s,1) = {'AML'};
    else
        actual(s,1)={'ALL'};
    end
end
percentage=(abs(d_test-out))*100;
DesiredValues_ActualValues_PercentageError=[d_test out percentage]
DesiredNames_ActualNames=[d_result actual]
