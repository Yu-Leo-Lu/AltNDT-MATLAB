function VisualizePredictionPlasmaSphereByWeights(NeuralNet,ParameterLabels,EnvironParam,nLayer,nZone,procFcnsInput,settingsXTrain,Stat,titleStr,Transition,isColorBar)
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
NNParam=zeros(1, length(ParameterLabels));

statLRange = [Stat.LRange]';
statnDataPoint = sqrt([Stat.nDataPoint])/100;
statMean = [Stat.DensityMean];
statStd = [Stat.DensitySTD];

% Evaluate electron density
for iLayer=1:nLayer
    for iZone=1:nZone
        for k=1:length(ParameterLabels)
            switch ParameterLabels{k}
                case 'l'
                    NNParam(k)=L(iLayer);
                case 'mlt'
                    % why /15? convert MLT from degree 0-360 to time 0-24
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
        
        try
            % by SGD
            yPred=predict(NeuralNet,NNParam);
        catch
            % by LM
            yPred = NeuralNet(NNParam');
        end
        % model is weighted, then stat scaled:
        iL = find(statLRange(:,1)<L(iLayer) & statLRange(:,2)>L(iLayer));
        yPred = yPred*nanmean(statStd(iL))*nanmean(statnDataPoint(iL)) + nanmean(statMean(iL));
        Density(iZone,iLayer) = yPred;
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
caxis([0,4]);
h=colorbar;
myMap = jet;

% readjust scales of jet map:
myMap((256-32*2):256, 1) = linspace(1,0.75,32*2+1);
myMap((256-32*2):256, 2) = 0;
myMap((256-32*2):256, 3) = 0;

myMap((256-32*3):(256-32*2), 1) = 1;
myMap((256-32*3):(256-32*2), 2) = linspace(0.5,0,32*1+1);

myMap((256-32*5):(256-32*3), 1) = 1;
myMap((256-32*5):(256-32*3), 2) = linspace(1,0.5,32*2+1);
myMap((256-32*5):(256-32*3), 3) = 0;

myMap((256-32*6):(256-32*5), 1) = linspace(0,1,32*1+1);
myMap((256-32*6):(256-32*5), 2) = 1;
myMap((256-32*6):(256-32*5), 3) = linspace(1,0,32*1+1);

myMap((256-32*7-16):(256-32*6), 1) = 0;
myMap((256-32*7-16):(256-32*6), 2) = linspace(0,1,32*1+1+16);
myMap((256-32*7-16):(256-32*6), 3) = 1;

myMap(9:16, 1) = linspace(1,0,8);
myMap(9:16, 2) = linspace(1,0,8);
myMap(9:16, 3) = 1;

myMap(1:8, 1) = 1;
myMap(1:8, 2) = 1;
myMap(1:8, 3) = 1;

% % update blue to white at the end of color bar
% b2w = 31;
% % update blue
% myMap(1:b2w, 3) = linspace(1,myMap(b2w+1,3) ,b2w);
% % update green
% myMap(1:b2w, 1) = linspace(1,0,b2w);
% % update red
% myMap(1:b2w, 2) = linspace(1,0,b2w);

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
ax = gca;
ax.TitleFontSizeMultiplier  = 2;
% title(['Predicted Plasma Density Data ', titleStr]);

return
end



        

