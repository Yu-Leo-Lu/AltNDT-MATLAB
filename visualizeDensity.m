function visualizeDensity(DensityTest, DensityPred, LTest, MLTTest, Stat, nameStr)

if ~exist('nameStr', 'var')
    nameStr = '';
end

NormalizedError = DensityTest - DensityPred;
for iCell = 1:length(Stat)
    ind=find((LTest>=Stat(iCell).LRange(1))&(LTest<Stat(iCell).LRange(2))&...
            (MLTTest>=Stat(iCell).LonRange(1)/15)&(MLTTest<Stat(iCell).LonRange(2)/15));
    if length(ind)>1
        cellSTD = Stat(iCell).DensitySTD;
        NormalizedError(ind) = NormalizedError(ind)/cellSTD;
    end
end

figure
histogram(NormalizedError);
title('Histogram of Normalization Error', nameStr);

figure
polarscatter(MLTTest*15*pi/180, LTest,[], NormalizedError)
colorbar
title('Normalization Error', nameStr);