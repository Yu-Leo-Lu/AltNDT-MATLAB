function VisualizePcaByWeights(NeuralNet,ParameterLabels,EnvironParams,nLayer,nZone,procFcnsInput,settingsXTrain,Stat,titleStr,Transition,isColorBar)
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

if isempty(titleStr)
    titleStr = '';
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
NNParam=zeros(2, length(ParameterLabels));
yPred=zeros(2,1);

statLRange = [Stat.LRange]';
statnDataPoint = [Stat.nDataPoint];
statMean = [Stat.DensityMean];
statStd = [Stat.DensitySTD];

% Evaluate electron density
for iLayer=1:nLayer
    for iZone=1:nZone
        for iRow = 1:2
            for k=1:length(ParameterLabels)
                switch ParameterLabels{k}
                    case 'l'
                        NNParam(iRow,k)=L(iLayer);
                    case 'mlt'
                        % why /15? convert MLT from degree 0-360 to time 0-24
                        NNParam(iRow,k)=MLT(iZone)/15;
                    case 'smlt'
                        NNParam(iRow,k)=sin(MLT(iZone)*pi/180);
                    case 'cmlt'
                        NNParam(iRow,k)=cos(MLT(iZone)*pi/180);
                    otherwise
                        NNParam(iRow,k)=EnvironParams(iRow,k);
                end
            end
            
            NNParam(iRow,:) = preProcessApply(NNParam(iRow,:), procFcnsInput, settingsXTrain);
            try
                % by SGD
                yPred(iRow)=predict(NeuralNet,NNParam(iRow,:));
            catch
                % by LM
                yPred(iRow) = NeuralNet(NNParam(iRow,:)');
            end
            % model is weighted, then stat scaled:
            iL = find(statLRange(:,1)<L(iLayer) & statLRange(:,2)>L(iLayer));
            
            % weight scaled proportional to density, not just # of data pts
            if statLRange(iL,2)~= inf
                ringArea = pi*(statLRange(iL,2)^2 - statLRange(iL,1)^2);
            else
                % 6.5697 = max earth radii recorded in XTrain, max(XTrain(:,5))
                ringArea = pi*(6.5697^2 - statLRange(iL,1)^2);
            end
            ringDensity = statnDataPoint(iL)/ringArea;
            ringWeight = sqrt(ringDensity)/10;
    
            
            
            yPred(iRow) = yPred(iRow)*nanmean(statStd(iL))*nanmean(ringWeight) + nanmean(statMean(iL));
        end
        Density(iZone,iLayer) = yPred(2)-yPred(1);
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
DensityVector=reshape(Density,nZone*nLayer,1);
%
% figure;
patch('Faces',Faces,'Vertices',Vertices,'FaceVertexCData',DensityVector,...
    'EdgeColor','none','FaceColor','flat');
%
% Plot logitude and altitude lines and the globe.
%
hold on
Theta=360*[0:1000]/1000;

for iL=1:7
%     x=iL*cos(Theta*pi/180);
%     y=iL*sin(Theta*pi/180);
%     plot(x,y,'-k');
if mod(iL,2)==0
    text(iL+0.1,0.15,num2str(iL));
end
%     text(-iL+0.1,0.15,num2str(iL));
end

% for iL=1:7
%     x=iL*cos(Theta*pi/180);
%     y=iL*sin(Theta*pi/180);
%     plot(x,y,'-k');
%     text(iL+0.1,0.15,num2str(iL));
%     text(-iL+0.1,0.15,num2str(iL));
% end
% for iLon=0:45:315
%     x=[0,7]*cos(iLon*pi/180);
%     y=[0,7]*sin(iLon*pi/180);
%     plot(x,y,'-k');
%     text(x(2)*(1.05),y(2)*(1.05),num2str(iLon));
% end
x=cos((-90:90)*pi/180);
y=sin((-90:90)*pi/180);
patch(x,y,[0,0,0]);
x=cos((90:270)*pi/180);
y=sin((90:270)*pi/180);
patch(x,y,[1,1,1]);
caxis([-2,2]);
h=colorbar;
myMap = jet;

% readjust scales of jet map:
% myMap((256-32*2):256, 1) = linspace(1,0.75,32*2+1);
% myMap((256-32*2):256, 2) = 0;
% myMap((256-32*2):256, 3) = 0;
% 
% myMap((256-32*3):(256-32*2), 1) = 1;
% myMap((256-32*3):(256-32*2), 2) = linspace(0.5,0,32*1+1);
% 
% myMap((256-32*5):(256-32*3), 1) = 1;
% myMap((256-32*5):(256-32*3), 2) = linspace(1,0.5,32*2+1);
% myMap((256-32*5):(256-32*3), 3) = 0;
% 
% myMap((256-32*6):(256-32*5), 1) = linspace(0,1,32*1+1);
% myMap((256-32*6):(256-32*5), 2) = 1;
% myMap((256-32*6):(256-32*5), 3) = linspace(1,0,32*1+1);
% 
% myMap(1:(256-32*6), 1) = 0;
% myMap(1:(256-32*6), 2) = linspace(0,1,64);
% myMap(1:(256-32*6), 3) = 1;

colormap(myMap);
ylabel(h,'Log(density)');
axis off

if ~isempty(Transition)
    indTrans=nLayer*ones(nZone,1);
    for iZone=1:nZone
        ind=find(Density(iZone,:)>Transition);
        if isempty(ind)
            indTrans(iZone)=1;
        end
        indTrans(iZone)=ind(end);
    end
    x=cos(2*pi*[0:nZone-1]/nZone).*(1+5*indTrans'/nLayer);
    y=sin(2*pi*[0:nZone-1]/nZone).*(1+5*indTrans'/nLayer);
    
    plot([x,x(1)],[y,y(1)],'k','LineWidth',[2])
end

if isColorBar == 0
    colorbar off
end

axis off
title(titleStr);
% title(['Predicted Plasma Density Data ', titleStr]);

return
end



        

