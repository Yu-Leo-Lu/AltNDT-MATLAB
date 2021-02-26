function [AIC, BIC] = infoCrit(SSE, n, p)

% SSE: Sum of Squared Errors for the training set
% n = length(net.trainInd); Number of training cases
% p = length(getwb(net)); Number of parameters (weights and biases)
% Schwarz's Bayesian criterion (or BIC) (Schwarz, 1978)
% BIC = n * log(SSE/n) + p * log(n)
% Akaike's information criterion (Akaike, 1969)
% AIC = n * log(SSE/n) + 2 * p

% Then the quantity exp((AICmin − AICi)/2) can be interpreted as 
% being proportional to the probability that the ith model 
% minimizes the (estimated) information loss.[5]

AIC = n * log(SSE/n) + 2 * p;
BIC = n * log(SSE/n) + p * log(n);

end

