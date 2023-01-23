function [positiveCellInds, negativeCellInds] = identifyPositiveCells(image, ROIs, thresholds)
    positiveCellInds = [];
    negativeCellInds = [];
    for i = 1:length(ROIs)
        coor = ROIs{i};
        img_mask = zeros(size(image));
        for j = 1:size(coor,2)
            img_mask(coor(2,j),coor(1,j)) = 1;
        end
        img_mask = imfill(img_mask, 'holes');
        if quantile(nonzeros(double(image).*img_mask), 0.8) > thresholds(1)
            positiveCellInds = [positiveCellInds; i];
        elseif quantile(nonzeros(double(image).*img_mask), 0.8) < thresholds(2) 
            negativeCellInds = [negativeCellInds; i];
        end
    end
end