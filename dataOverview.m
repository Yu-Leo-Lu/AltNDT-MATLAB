[P,inds_train, inds_test] = loadPINE();

l = P.data_all.X(:,5);
mlt = P.data_all.X(:,6);
t = P.data_all.t;

figure
hist_l = histogram(l);
title('histogram of l')

figure
hist_mlt = histogram(mlt);
title('histogram of mlt')

rad = mlt*2*pi/24;
figure
polarscatter(l, rad, 1, t)
colorbar
pax = gca;
pax.ThetaDir = 'clockwise';
pax.ThetaZeroLocation = 'bottom';
pax.ThetaTick = [0,2,4,6,8,10,12,14,16,18,20,22,24];
pax.ThetaLim = [0,24];