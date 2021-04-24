function [Stat]=getPlasmaSphereStatsL(L,Density,varargin)
%
% Description: Evaluate regional statistics of electron density over
%   plasmaspheric region. The region is define on equatorial plane in polar
%   coordinates radius L and longitudinal angle in MLT (magnetic local
%   time).
%
nargin_default=3;
%
nAlt_Layers=20;
minCellSize=10;
if nargin>nargin_default
    if mod(nargin-nargin_default,2)~=0
        error('getPlasmaSphereStats: Optional input arguments must be Property-Value pairs.');
    end
    for k=1:2:nargin-nargin_default
        if ~ischar(varargin{k})
            error('getPlasmaSphereStats: Optional input arguments must starts with property name.');
        end
        switch varargin{k}
            case 'nAlt_Layers'
                nAlt_Layers=varargin{k+1};
            case 'minCellSize'
                minCellSize=varargin{k+1};
        end
    end
end
% EarthRadius=6378;
minL=min(L);
maxL=max(L);
Layers=minL+(maxL-minL)*[0:nAlt_Layers-1;1:nAlt_Layers]/nAlt_Layers;
Layers(2,end)=inf;
nCell=ones(nAlt_Layers,1);
%
% Convert minCellSize from degree to km.
%
% minCellSize=2*pi*EarthRadius*minL*minCellSize/360;
% for iLayer=1:nAlt_Layers
%     nCell(iLayer)=floor(2*pi*EarthRadius*Layers(1,iLayer)/minCellSize);
% end
totalCell=sum(nCell);
Stat=struct('LRange',cell(totalCell,1),'LonRange',cell(totalCell,1),'DensityMean',cell(totalCell,1),...
    'DensitySTD',cell(totalCell,1),'nDataPoint',cell(totalCell,1));
iCell=0;
for iLayer=1:nAlt_Layers
    for iLon=1:nCell(iLayer)
        iCell=iCell+1;
        Stat(iCell).LRange=Layers(:,iLayer);
        Stat(iCell).LonRange=[iLon-1;iLon]*360/nCell(iLayer);
        ind=find((L>=Stat(iCell).LRange(1))&(L<Stat(iCell).LRange(2)));
        Stat(iCell).nDataPoint=length(ind);
        Stat(iCell).DensityMean=nanmean(Density(ind));
        Stat(iCell).DensitySTD=nanstd(Density(ind));
    end
end
return
end
    
