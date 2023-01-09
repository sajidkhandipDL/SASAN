name_unColImg = 'sample 3 hemangioma';
image_color = imread([name_unColImg '.png']);
mkdir(name_unColImg);
imwrite(image_color, [name_unColImg '/' '0 clor image.png']);
image_gray = im2double(rgb2gray(image_color));
image_diff = 1 - image_gray;
imwrite(image_diff, [name_unColImg '/' '0 image negative.png']);
image_diff = image_diff>0;
imwrite(image_diff, [name_unColImg '/' '0 binary image.png']);

image_labels = bwlabel(image_diff);
coloredLabels = label2rgb (image_labels, 'hsv', 'k', 'shuffle');
imwrite(coloredLabels, [name_unColImg '/' '0 colored connectivity.png']);

unique_labels = unique(image_labels);
index_curr = 0;
for i = 1:length(unique_labels)-1
    [row_lab, col_lab] = find(image_labels==i);
    min_row = min(row_lab);
    min_col = min(col_lab);
    max_row = max(row_lab);
    max_col = max(col_lab);
    patch_curr = image_color(min_row:max_row, min_col:max_col,:);
    if (((max_col-min_col)*(max_row-min_row))>500)
        imwrite(patch_curr, [name_unColImg '/' num2str(index_curr) '.png']);
        index_curr = index_curr+1;
    end
    
end
