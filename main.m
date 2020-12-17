function main
image=imread("training_images/train11.jpg");
%enhance edges
kernel = -1*ones(3);
kernel(2,2) =100;
enhancedImage = imfilter(image, kernel);
%enahcne brightness and contrast
image=imadjust(image,[0,0.9],[0 1]);
copy=imadjust(image,[0.1,0.9],[0 1]);
%threshold to seperate red and blue blocks
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
k=2;
for i =1:length(bstats)
    BB = bstats(i).BoundingBox;
    if bstats(i).Area>15000 && bstats(i).Area>50000 
    subImage = imcrop(image,BB);
    [bboxes, scores] = detect(detector,subImage);
     if length(scores)
        rectangle('Position', [BB(1),BB(2),BB(3),BB(4)],'EdgeColor','r','LineWidth',k) ;
        blue=blue+1;
        k=k+1;
        display(length(scores));
     %end
    end
    end
end  
%Red blocks   
rred=~red;
image=copy;
image(rred)=255;
figure(3),imshow(image);
bstats = regionprops(red,'all');
red=0;
figure(4),imshow(copy)
k=1;
for i =1:length(bstats)
    BB = bstats(i).BoundingBox;
    if bstats(i).Area>10000 && bstats(i).Area<60000
    subImage = imcrop(image,BB);
    [bboxes, scores] = detect(rdetector,subImage);
     if length(scores)<10
        rectangle('Position', [BB(1),BB(2),BB(3),BB(4)],'EdgeColor','r','LineWidth',k) ;
        red=red+1;
        k=k+1;
        display(length(scores))
     end
    end
end
end

