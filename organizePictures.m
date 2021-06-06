c=get(gcf, 'children')
delete(c(1))
get(c(10))
get(c(10), 'position')
set(c(1), 'position', [0.6184    0.5482    0.24    0.1577])


c=get(gcf, 'children')
for i = 1:length(c)
    set(c(i), 'TitleFontSizeMultiplier', 2.5)
end