function VisualizePlasmaSPhereStats(Stat, nameStr)
%
% Description: Create polar plot for statistics of plasmasphere density.
% Usage: VisualizePlasmaSPhereStats(Stat,FieldName)
%
if ~exist('nameStr', 'var')
    nameStr = '';
end

nCell=length(Stat);
Faces=zeros(nCell,8);
Vertices=zeros(8*nCell,2);
iVertices=0;
for iCell=1:nCell
    LonRange=Stat(iCell).LonRange(1)+(Stat(iCell).LonRange(2)-Stat(iCell).LonRange(1))*(0:3)/3;
    Vertices(iVertices+(1:4),1)=Stat(iCell).LRange(1)*cos(LonRange*pi/180);
    Vertices(iVertices+(8:-1:5),1)=Stat(iCell).LRange(2)*cos(LonRange*pi/180);
    Vertices(iVertices+(1:4),2)=Stat(iCell).LRange(1)*sin(LonRange*pi/180);
    Vertices(iVertices+(8:-1:5),2)=Stat(iCell).LRange(2)*sin(LonRange*pi/180);
    Faces(iCell,:)=iVertices+(1:8);
    iVertices=iVertices+8;
end
Field={'DensityMean','DensitySTD','nDataPoint'};
for iPlot=1:3
    figure;
    colormap('jet');
    set(gcf,'Position',[270,160,850,780],'Color','w');
    Value=[Stat(:).(Field{iPlot})];
    patch('Faces',Faces,'Vertices',Vertices,'FaceVertexCData',Value',...
        'EdgeColor','none','FaceColor','flat');
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
    if strcmp(Field{iPlot}, 'DensityMean')
        caxis([0,3.5]);
    end
    h=colorbar;
    ylabel(h,Field{iPlot});
    axis off
    title([nameStr,' Statistics of Plasma Density Data: ',Field{iPlot}]);
end
return
end