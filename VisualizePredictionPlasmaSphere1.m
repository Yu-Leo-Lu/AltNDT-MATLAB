function VisualizePredictionPlasmaSphere1(NeuralNet,ParameterLabels,EnvironParam,nLayer,nZone,varargin)
%
% Description: Generate a 2D polar plot of electron density in the
%      plasmasphere predicted by a neural network.
%
% Usage:
%       VisualizePredictionPlasmaSphere(NeuralNet,ParameterLabels,EnvironParam)
%       VisualizePredictionPlasmaSphere(NeuralNet,ParameterLabels,EnvironParam,'DensityRange',range)
%       VisualizePredictionPlasmaSphere(NeuralNet,ParameterLabels,EnvironParam,'DensityRange',range,'Transition',trans)
%   Inputs:
%       NeuralNet        : Neural network object.
%       ParameterLabels  : Cell array of character strings giving the name 
%                          of input variables forthe neural network. The 
%                          list is ordered as EnvironParam below except it
%                          also contains L and MLT or polar coordinates
%                          replacing MLT.
%       EnvironParam     : Parts of input parameter for the neural network.
%       nLayer           : Number altitude layers from 1 to 6 au. 
%       nZone            : Number of longitude cells between 0 and 360
%                          degree.
%       range            : Specify the range of density for the plotting.
%
nargin_default=5;
if nargin>nargin_default
    if mod(nargin-nargin_default,2)~=0
        error('VisualizePredictionPlasmaSphere: Optional parameters must be property-value pair');
    end
    for k=1:2:nargin-nargin_default
        if ~ischar(varargin{k})
            error('VisualizePredictionPlasmaSphere: Property name must be a string');
        end
        switch varargin{k}
            case 'DensityRange'
                DensityRange=varargin{k+1};
            case 'Transition'
                Transition=varargin{k+1};
            otherwise
                error(['VisualizePredictionPlasmaSphere: Unknown property:',varargin{k}]);
        end
    end
end
Density=NaN(nZone,nLayer);
L=1+5*[0.5:1:nLayer-0.5]/nLayer;
MLT=360*[0.5:1:nZone-0.5]/nZone;
indL=find(strcmp(ParameterLabels,'L')==1);
indMLT=find(strcmp(ParameterLabels,'MLT')==1);
if isempty(indMLT)
    indSML=find(strcmp(ParameterLabels,'SML')==1);
    indCML=find(strcmp(ParameterLabels,'CML')==1);
end
NNParam=zeros(length(ParameterLabels),1);
%
% Evaluate electron density
%
for iLayer=1:nLayer
    for iZone=1:nZone
        indFixVal=0;
        for k=1:length(ParameterLabels)
            switch ParameterLabels{k}
                case 'L'
                    NNParam(k)=L(iLayer);
                case 'MLT'
                    NNParam(k)=MLT(iZone);
                case 'SML'
                    NNParam(k)=sin(MLT(iZone)*15*pi/180);
                case 'CML'
                    NNParam(k)=cos(MLT(iZone)*15*pi/180);
                otherwise
                    indFixVal=indFixVal+1;
                    NNParam(k)=EnvironParam(indFixVal);
            end
        end
        Density(iZone,iLayer)=predict(NeuralNet,NNParam);
    end
end
%
% Define the polygons.
%
Vertices=NaN(nZone,nLayer+1,2);
Vertices(:,:,1)=cos(2*pi*[0:nZone-1]'/nZone)*(1+5*[0:nLayer]/nLayer);
Vertices(:,:,2)=sin(2*pi*[0:nZone-1]'/nZone)*(1+5*[0:nLayer]/nLayer);
Vertices=reshape(Vertices,nZone*(nLayer+1),2);
Faces=NaN(nZone,nLayer,5);
Faces(:,:,1)=[1:nZone]'*ones(1,nLayer)+ones(nZone,1)*[0:nLayer-1]*nZone;
Faces(:,:,2)=[1:nZone]'*ones(1,nLayer)+ones(nZone,1)*[1:nLayer]*nZone;
Faces(:,:,3)=[2:nZone,1]'*ones(1,nLayer)+ones(nZone,1)*[1:nLayer]*nZone;
Faces(:,:,4)=[2:nZone,1]'*ones(1,nLayer)+ones(nZone,1)*[0:nLayer-1]*nZone;
Faces(:,:,5)=[1:nZone]'*ones(1,nLayer)+ones(nZone,1)*[0:nLayer-1]*nZone;
Faces=reshape(Faces,nZone*nLayer,5);
Density=reshape(Density,nZone*nLayer,1);
%
figure;
patch('Faces',Faces,'Vertices',Vertices,'FaceVertexCData',Density,...
    'EdgeColor','none','FaceColor','flat');
if exist('DensityRange','var')
    caxis(DensityRange)
end
%
% Plot logitude and altitude lines and the globe.
%
hold on
Theta=360*[0:1000]/1000;
for iL=1:7
    x=iL*cos(Theta*pi/180);
    y=iL*sin(Theta*pi/180);
    plot(x,y,'-k');
    text(iL+0.1,0.15,num2str(iL));
    text(-iL+0.1,0.15,num2str(iL));
end
for iLon=0:45:315
    x=[0,7]*cos(iLon*pi/180);
    y=[0,7]*sin(iLon*pi/180);
    plot(x,y,'-k');
    text(x(2)*(1.05),y(2)*(1.05),num2str(iLon));
end
x=cos((-90:90)*pi/180);
y=sin((-90:90)*pi/180);
patch(x,y,[0,0,0]);
x=cos((90:270)*pi/180);
y=sin((90:270)*pi/180);
patch(x,y,[1,1,1]);
h=colorbar;
ylabel(h,'Log(density)');
axis off
if exist('Transition','var')
    indTrans=nLayer*ones(nZone,1);
    for iZone=1:nZone
        ind=find(Density(iZone,:)>Transition);
        if isempty(ind)
            indTrans(iZone)=1;
        end
        indTrans(iZone)=ind(end);
    end
    x=cos(2*pi*[0:nZone-1]/nZone).*(1+5*indTrans/nLayer);
    y=sin(2*pi*[0:nZone-1]/nZone).*(1+5*indTrans/nLayer);
    plot([x,x(1)],[y,y(1)],'k','LineWidth',[2])
end
title(['Predicted Plasma Density Data']);
return
end



        

