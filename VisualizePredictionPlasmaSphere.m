function VisualizePredictionPlasmaSphere(NeuralNet,ParameterLabels,EnvironParam,nLayer,nZone,procFcnsInput,settingsXTrain,Stat,nameStr)
%
% Description: Generate a 2D polar plot of electron density in the
%      plasmasphere predicted by a neural network.
%
% Usage:
%       VisualizePredictionPlasmaSphere(NeuralNet,ParameterLabels,EnvironParam)
%       VisualizePredictionPlasmaSphere(NeuralNet,ParameterLabels,EnvironParam,'DensityRange',range)
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

if isempty(nameStr)
    nameStr = '';
end

Density=NaN(nZone,nLayer);
L=1+5*[0.5:1:nLayer-0.5]/nLayer;
% MLT: 0-360 in degree
MLT=360*[0.5:1:nZone-0.5]/nZone;
indL=find(strcmp(ParameterLabels,'l')==1);
indMLT=find(strcmp(ParameterLabels,'mlt')==1);
if isempty(indMLT)
    indSML=find(strcmp(ParameterLabels,'smlt')==1);
    indCML=find(strcmp(ParameterLabels,'cmlt')==1);
end
NNParam=zeros(1, length(ParameterLabels));
%
% Evaluate electron density
%
for iLayer=1:nLayer
    for iZone=1:nZone
        for k=1:length(ParameterLabels)
            switch ParameterLabels{k}
                case 'l'
                    NNParam(k)=L(iLayer);
                case 'mlt'
                    % convert MLT from degree 0-360 to time 0-24
                    NNParam(k)=MLT(iZone)/15;
                case 'smlt'
                    NNParam(k)=sin(MLT(iZone)*pi/180);
                case 'cmlt'
                    NNParam(k)=cos(MLT(iZone)*pi/180);
                otherwise
                    NNParam(k)=EnvironParam(k);
            end
        end
        NNParam = preProcessApply(NNParam, procFcnsInput, settingsXTrain);
        yPred=predict(NeuralNet,NNParam);
        
        % if the model is Stat scaled:
        if ~isempty(Stat)
            statMean = [Stat.DensityMean];
            statStd = [Stat.DensitySTD];
            statnData = [Stat.nDataPoint];
            statLRange = [Stat.LRange]';
            statLonRange = [Stat.LonRange]';
            iL = find(statLRange(:,1)<L(iLayer) & statLRange(:,2)>L(iLayer));
            iLMLT = find(statLRange(:,1)<L(iLayer) & statLRange(:,2)>L(iLayer) &...
                statLonRange(:,1)<MLT(iZone) & statLonRange(:,2)>MLT(iZone));
            if isempty(iLMLT) || statnData(iLMLT)<=1
                Density(iZone,iLayer) = yPred*nanmean(statStd(iL)) + nanmean(statMean(iL));
            else
                Density(iZone,iLayer) = yPred*statStd(iLMLT) + statMean(iLMLT);
            end

        else
            Density(iZone,iLayer) = yPred;
        end
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
% figure;
patch('Faces',Faces,'Vertices',Vertices,'FaceVertexCData',Density,...
    'EdgeColor','none','FaceColor','flat');
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
caxis([0,4]);
h=colorbar;
myMap = jet;

% update blue to white at the end of color bar
b2w = 31;
% update blue
myMap(1:b2w, 3) = linspace(1,myMap(b2w+1,3) ,b2w);
% update green
myMap(1:b2w, 1) = linspace(1,0,b2w);
% update red
myMap(1:b2w, 2) = linspace(1,0,b2w);

% update yellow to cyan transition
% 160 row is yellow 1,1,0
% 96 row is cyan 0,1,1
% there are 64 rows between them, 
% update the last quater of 64 rows (16 rows) to grey
% update the last quater of 16 grey rows to black
% myMap(160-16:159, :) = 0.5;
% myMap(160-8:160-4, :) = 0;
colormap(myMap);
ylabel(h,'Log(density)');
axis off
title(['Predicted Plasma Density Data ', nameStr]);
return
end



        

