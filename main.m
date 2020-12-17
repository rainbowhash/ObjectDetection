function main
image=imread("training_images/train04.jpg");
image=imadjust(image,[0.1,0.9],[0 1]);
copy=imadjust(image,[0.3,1],[0 1]);
red=createMaskRed(image);
blue=createMaskBlue(image);
inv=~blue;
image(inv)=255;
figure(1),imshow(image);
persistent s detector 
if isempty(detector)
  	s = coder.load('detectorblue.mat');
	detector = acfObjectDetector(s.Classifier,s.TrainingOptions);
end
persistent y rdetector
if isempty(rdetector)
  	y = coder.load('detectorRed.mat');
	rdetector = acfObjectDetector(y.Classifier,y.TrainingOptions);
end
%blue blocks
bstats = regionprops(blue,'all');
blue=0;
figure(2),imshow(copy)
for i =1:length(bstats)
    BB = bstats(i).BoundingBox;
    if bstats(i).Area>20000 
    subImage = imcrop(image,BB);
    [bboxes, scores] = detect(detector,subImage);
     if length(scores)
        rectangle('Position', [BB(1),BB(2),BB(3),BB(4)],'EdgeColor','r','LineWidth',2) ;
        blue=blue+1;
        display(length(scores))
     %end
    end
    end
end  
    
rred=~red;
image=copy;
image(rred)=255;
figure(3),imshow(image);
bstats = regionprops(red,'all');
red=0;
figure(4),imshow(copy)
for i =1:length(bstats)
    BB = bstats(i).BoundingBox;
    if bstats(i).Area>1000 && bstats(i).Area<20000
    subImage = imcrop(image,BB);
    [bboxes, scores] = detect(rdetector,subImage);
     if length(scores)
        rectangle('Position', [BB(1),BB(2),BB(3),BB(4)],'EdgeColor','r','LineWidth',2) ;
        red=red+1;
        display(length(scores))
     end
    end
end
end
%[bboxes, scores] = detect(detector,image);


% for i = 1:length(scores)
%    annotation = sprintf('Confidence = %.1f',scores(i));
%    if scores(i)>95
%        image = insertObjectAnnotation(image,'rectangle',bboxes(i,:),annotation);
%    end
% end
% figure
% imshow(image)
% end

